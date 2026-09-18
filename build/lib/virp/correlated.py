"""
Tools for disordered crystal structures: sites where more than one
species (or a vacancy) shares a position, with occupancies summing to 1.

shap_motifs(): ranks "motifs" (named local coordination environments,
e.g. "Cu-S4" = a Cu atom with 4 S neighbors) by SHAP correlation with a
target property (e.g. formation energy).

sat_solver() / cluster_infeasible_rules(): given a disordered structure
and ranked motif rules, use a CP-SAT solver to pick one real occupant
per disordered site (preserving stoichiometry) that satisfies those
rules as well as possible. sat_solver enumerates every distinct
solution, up to a cap.

Key terms:
- disorder group: disordered sites close enough together to be the same
  physical slot -- at most one can be real. `method` picks the
  closeness test; see _build_sat_context.
- site network: a disorder group connected to its own periodic image --
  an infinite chain/sheet/framework, not a finite slot. See
  _detect_group_networks.
- closed-world motif: a species missing from a motif's neighbor counts
  must have exactly zero of that neighbor, not "don't care".
"""

# --- shap_motifs (motif importance ranking) ---
import random
import secrets
from pathlib import Path
from typing import Optional, Sequence, Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --- sat_solver / cluster_infeasible_rules (the CP-SAT disorder solver) ---
import os
import re
import shutil
from dataclasses import dataclass
from collections import Counter
from pymatgen.core import Element, Structure
from pymatgen.core.local_env import CrystalNN
from ortools.sat.python import cp_model

# --- solver diagnostics: one picture per disorder group, pie-wedged by
# candidate species via ASE's atoms.info["occupancy"] + atoms.get_tags()
from ase import Atoms
from ase.visualize.plot import plot_atoms
import ase.io.utils as _ase_io_utils
from ase.data import atomic_numbers as _ase_atomic_numbers

# ASE draws H white by default, invisible against the page and the white
# "vacancy" wedge of a partially-occupied site -- recolor it once.
_ase_io_utils.default_colors[_ase_atomic_numbers["H"]] = np.array([0.75, 0.75, 0.75])


SAVE_KWARGS = dict(dpi=300, bbox_inches="tight")

# Outer bound (Å) for the pairwise-distance scan; must exceed the
# largest radius_threshold * (r_i + r_j) you expect to apply.
_GROUP_SEARCH_RADIUS = 4.0

# "radii" method: group sites closer than this * (r_i + r_j). 1.0 =
# group when their atomic radii would literally overlap.
DEFAULT_RADIUS_THRESHOLD = 1.0

# "histogram" method: the disorder-split/real-bond boundary is the
# first distance gap whose ratio to the previous distance reaches this.
# See _auto_group_cutoff.
DEFAULT_JUMP_RATIO = 1.7


# ==============================================================
# Ancillary Functions (shap)
# ==============================================================

def _merge_df(df_motifs: pd.DataFrame, df_energies: pd.DataFrame) -> pd.DataFrame:
    key_motifs, key_energies = df_motifs.columns[0], df_energies.columns[0]

    names_motifs = set(df_motifs[key_motifs])
    names_energies = set(df_energies[key_energies])
    if names_motifs != names_energies:
        missing = names_motifs.symmetric_difference(names_energies)
        raise ValueError(
            "Name values do not match! Check the same material is featured "
            f"in motif and energy data. Mismatched entries: {sorted(missing)}"
        )

    df = df_motifs.merge(df_energies, left_on=key_motifs, right_on=key_energies, how="left")
    if key_motifs != key_energies:
        df = df.drop(columns=key_energies)
    return df


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """First column = identifier (dropped), last = target property,
    middle columns = motif-count features."""
    y = df.iloc[:, -1]
    X = df.iloc[:, 1:-1]
    X = X.loc[:, X.nunique() > 1]  # drop constant columns
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"Structures : {len(df)}")
    print(f"Features   : {X.shape[1]}")
    return X, y


def _resolve_seed(random_state: Optional[int]) -> int:
    """None draws a fresh random seed, an int pins one. Always printed,
    so a "random" run can be repeated later."""
    seed = secrets.randbits(32) if random_state is None else random_state
    print(f"Random seed  : {seed}")
    return seed


def _train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def _report_performance(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    pred = model.predict(X_test)
    print("\nModel performance")
    print("-----------------")
    print("R²  :", r2_score(y_test, pred))
    print("MAE :", mean_absolute_error(y_test, pred))


def _pearson_corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between matching columns of `a` and `b`,
    vectorized across all columns."""
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    numerator = (a_c * b_c).sum(axis=0)
    denominator = np.sqrt((a_c**2).sum(axis=0) * (b_c**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator != 0, numerator / denominator, 0.0)


def _save_fig(path: str) -> None:
    plt.savefig(path, **SAVE_KWARGS)
    plt.close()


def _plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    shap.plots.violin(
        shap.Explanation(values=shap_values, data=X.to_numpy(), feature_names=X.columns),
        show=False,
    )
    _save_fig(out_path)


def _plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    _save_fig(out_path)


def _build_importance_table(X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """One row per motif: MeanAbsSHAP (importance) and Correlation
    (which way it pushes the target). Correlation is MeanAbsSHAP signed
    by the Pearson correlation's direction, not a raw Pearson r, which
    is noisy on these mostly-zero columns."""
    correlations = _pearson_corr_columns(X.to_numpy(dtype=float), shap_values)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    signed_importance = np.sign(correlations) * mean_abs_shap
    return pd.DataFrame(
        {"Motif": X.columns, "MeanAbsSHAP": mean_abs_shap, "Correlation": signed_importance}
    ).sort_values("MeanAbsSHAP", ascending=False)


def _plot_correlation_bar(importance: pd.DataFrame, out_path: str) -> None:
    corr_df = importance.sort_values("Correlation")
    colors = np.where(corr_df["Correlation"] > 0, "red", "green")
    n = len(corr_df)
    fig_height = max(3, n * 0.15)
    plt.figure(figsize=(8, fig_height))
    plt.barh(corr_df["Motif"], corr_df["Correlation"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Signed SHAP Importance (sign = feature–SHAP correlation)", fontsize=14)
    plt.ylabel("Motif", fontsize=14)
    plt.xticks(fontsize=12)
    # Many motifs packed into a fixed-height figure leave little vertical
    # room per bar -- a constant fontsize then overlaps. Shrink the
    # y-axis (motif name) font to the space actually available per row.
    label_fontsize = min(12, max(4, (fig_height / n) * 72 * 0.7))
    plt.yticks(fontsize=label_fontsize)
    plt.tight_layout()
    _save_fig(out_path)


# ==============================================================
# Ancillary Functions (sat)
# ==============================================================

def _parse_candidates(rule_strings):
    """Parse 'Source-Motif' strings (e.g. "Cu-S4Fe2") into
    (source_species, motif_dict, annotation) triples, e.g.
    ("Cu", {"S": 4, "Fe": 2}, None). A bare species defaults to count 1.

    A trailing "(...)" local-symmetry tag, e.g. "Cl-Te (FO:7)", is split
    off into `annotation` (here "(FO:7)"; None if absent) rather than fed
    to the neighbor-count parser. It carries no constraint semantics of
    its own -- this codebase only ever matches raw neighbor counts, never
    true point-group/local symmetry -- but it DOES distinguish otherwise-
    identical motifs: "Cl-Te (FO:7)" and "Cl-Te (FO:12)" both constrain
    to {"Te": 1}, yet are kept and ranked as two separate candidates
    (never merged), and their annotation is carried through to every
    printed rule listing so the two stay visually distinct."""
    motif_token_re = re.compile(r"([A-Z][a-z]?)(\d*)")
    annotation_re = re.compile(r"^(.*?)\s*(\([^()]*\))$")
    parsed = []
    for raw in rule_strings:
        raw = raw.strip()
        ann_match = annotation_re.match(raw)
        core, annotation = (ann_match.group(1).strip(), ann_match.group(2)) if ann_match else (raw, None)
        if "-" not in core:
            raise ValueError(f"Malformed rule (expected 'Source-Motif'): {raw!r}")
        source, motif_str = core.split("-", 1)
        source, motif_str = source.strip(), motif_str.strip()
        motif, pos = {}, 0
        for m in motif_token_re.finditer(motif_str):
            species, count_str = m.groups()
            motif[species] = motif.get(species, 0) + (int(count_str) if count_str else 1)
            pos = m.end()
        if pos != len(motif_str):
            raise ValueError(f"Could not fully parse motif {motif_str!r} in rule: {raw!r}")
        parsed.append((source, motif, annotation))
    return parsed


def _dominant_species(site) -> str:
    """Majority-occupancy species at one site. Only for stand-in labels
    (geometry proxies, non-disordered sites) -- never to decide a
    disordered site's real species, which must stay a CP-SAT choice
    (member_species), or a minority occupant could never be selected."""
    occ_dict = site.species.as_dict()
    return max(occ_dict, key=occ_dict.get)


@dataclass
class _SATContext:
    """Structural/occupancy data the CP-SAT model is built from.

    Every site is either togglable (in member_species: a disorder-group
    member or an independently disordered site, gets a presence
    variable and a species choice if it has multiple candidates) or
    fixed (in species_of: occupancy 1, one species, no variable needed).

    Grouped sites also appear in group_of/groups, which drives the <=1
    (or ==1, if group_mandatory) exclusivity on a group's members --
    except a group in network_groups (connected to its own periodic
    image), which is exempt: only stoichiometry and motif rules apply.
    """
    structure: Structure
    labels: List[str]
    species_of: Dict[int, str]
    member_species: Dict[int, List[str]]
    togglable: Set[int]
    groups: Dict[int, List[int]]
    group_of: Dict[int, int]
    group_mandatory: Set[int]
    network_groups: Set[int]
    source_refs_by_species: Dict[str, List[int]]
    stoichiometry: Dict[str, int]
    neighbor_cache: Dict[int, List[dict]]
    method: str
    radius_threshold: Optional[float]
    species_radii_used: Optional[Dict[str, float]]
    group_cutoff: Optional[float]


# Union-find (disjoint-set): A-B and B-C unioned means A, B, C all group.

def _find(x, parent):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(x, y, parent):
    rx, ry = _find(x, parent), _find(y, parent)
    if rx != ry:
        parent[rx] = ry


def _union_find_groups(indices: Sequence[int], edges: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Transitively group `indices` connected by `edges`. Returns
    {group_id: sorted members}, for indices touched by >=1 edge only."""
    parent = {i: i for i in indices}
    touched: Set[int] = set()
    for i, j in edges:
        touched.add(i)
        touched.add(j)
        _union(i, j, parent)
    groups: Dict[int, List[int]] = {}
    for i in touched:
        groups.setdefault(_find(i, parent), []).append(i)
    return {gid: sorted(members) for gid, members in groups.items()}


def _apportion_targets(target: int, raws: Sequence[float]) -> List[int]:
    """Round `raws` (a species' raw occupancy sum within each of several
    site-type strata) to integers that sum EXACTLY to `target`, via
    largest-remainder (Hamilton) apportionment: floor every value, then
    hand the leftover units to whichever entries are closest to rounding
    up (or claw back from them, if `target` is under the floor-sum)."""
    floors = [int(r) for r in raws]
    remainder = target - sum(floors)
    fracs = [r - f for r, f in zip(raws, floors)]
    if remainder >= 0:
        order = sorted(range(len(raws)), key=lambda i: fracs[i], reverse=True)
        for i in order[:remainder]:
            floors[i] += 1
    else:
        order = sorted(range(len(raws)), key=lambda i: fracs[i])
        for i in order[:-remainder]:
            floors[i] -= 1
    return floors


def _reify_and(m: cp_model.CpModel, lits: List, name: str):
    """New bool var b with b <-> AND(lits)."""
    b = m.NewBoolVar(name)
    m.AddBoolAnd(lits).OnlyEnforceIf(b)
    m.AddBoolOr([l.Not() for l in lits]).OnlyEnforceIf(b.Not())
    return b


def _reify_or(m: cp_model.CpModel, lits: List, name: str):
    """New bool var b with b <-> OR(lits)."""
    b = m.NewBoolVar(name)
    m.AddBoolOr(lits).OnlyEnforceIf(b)
    m.AddBoolAnd([l.Not() for l in lits]).OnlyEnforceIf(b.Not())
    return b


def _hard_count(m: cp_model.CpModel, species_var, refs: List[int], sp: str, target: int) -> None:
    """Hard-constrain the number of `refs` realized as species `sp` to
    exactly `target`. Refs that are fixed constants (not real variables)
    are dropped -- they're already baked into `target`."""
    terms = [t for t in (species_var(r, sp) for r in refs) if not isinstance(t, int)]
    if terms:
        m.Add(sum(terms) == target)


def _sat_build_base(ctx: _SATContext):
    """Build the HARD part of the CP-SAT model: group exclusivity and
    stoichiometry (overall + per-site-type proportional). No motif rules
    yet -- shared by both rule modes in sat_solver (_sat_build_model for
    hard rules, _sat_build_model_soft for soft ones).

    pres[i]: boolean, is site i realized (every togglable site gets
    one). sp_choice[i, sp]: boolean species choice, for a site with more
    than one candidate species.

    Returns (m, pres, sp_choice, build_eq, presence_literal, label_for)
    -- the pieces a motif-rule layer builds on top of.
    """
    m = cp_model.CpModel()
    pres = {i: m.NewBoolVar(f"sel_{ctx.labels[i]}") for i in ctx.togglable}

    sp_choice = {}
    for i, candidates in ctx.member_species.items():
        if len(candidates) > 1:
            for sp in candidates:
                sp_choice[(i, sp)] = m.NewBoolVar(f"sp_{ctx.labels[i]}_{sp}")
            m.Add(sum(sp_choice[(i, sp)] for sp in candidates) == pres[i])

    for gid, members in ctx.groups.items():
        if gid in ctx.network_groups:
            continue  # periodic self-connection, not a finite slot -- no exclusivity
        total = sum(pres[i] for i in members)
        # Occupancies summing to ~1 means the slot is never really
        # vacant: exactly one member is realized, not merely at most one.
        m.Add(total == 1 if gid in ctx.group_mandatory else total <= 1)

    def species_var(ref, sp):
        """0/1 expression for 'ref is realized as species sp' -- a
        CP-SAT literal where that's a real choice, else a constant."""
        if ref in ctx.member_species:
            candidates = ctx.member_species[ref]
            if sp not in candidates:
                return 0
            return sp_choice[(ref, sp)] if len(candidates) > 1 else pres[ref]
        return 1 if ctx.species_of.get(ref) == sp else 0

    # Global count of sites realized as each species must hit its target.
    for sp, target in ctx.stoichiometry.items():
        _hard_count(m, species_var, ctx.source_refs_by_species.get(sp, []), sp, target)

    # Sites sharing the exact same occupancy signature (e.g. every
    # {"Fe":0.083,"Cu":0.25} site) are repeated instances of one site
    # type. Pin each type's own share too (via _apportion_targets, so
    # per-type integer targets sum exactly to the global one), or CP-SAT
    # could satisfy the global target by starving one type entirely.
    strata: Dict[tuple, List[int]] = {}
    for i in ctx.togglable:
        sig = tuple(sorted((sp, round(occ, 6)) for sp, occ in ctx.structure[i].species.as_dict().items()))
        strata.setdefault(sig, []).append(i)

    raw_by_species: Dict[str, List[Tuple[List[int], float]]] = {}
    for sig, members in strata.items():
        for sp, occ in dict(sig).items():
            raw_by_species.setdefault(sp, []).append((members, occ * len(members)))

    for sp, entries in raw_by_species.items():
        targets = _apportion_targets(ctx.stoichiometry.get(sp, 0), [raw for _, raw in entries])
        for (members, _), target in zip(entries, targets):
            _hard_count(m, species_var, members, sp, target)

    def any_species_var(ref):
        """0/1 expression for 'ref is realized as SOME species'."""
        return pres[ref] if ref in pres else 1

    def presence_literal(ref, sp):
        """(literal, always_true) for a constraint conditional on 'ref
        is species sp'. always_true means ref is a fixed site of exactly
        that species (constraint applies unconditionally). Uses the
        SPECIES CHOICE, not mere presence, at a multi-candidate site --
        pres[ref] is true for EITHER candidate, which would wrongly
        apply sp's motif to a site that resolved to a different species."""
        if ref in ctx.member_species:
            candidates = ctx.member_species[ref]
            if len(candidates) > 1:
                return sp_choice[(ref, sp)], False
            return pres[ref], False
        return None, True

    def label_for(ref):
        return ctx.labels[ref]

    def build_eq(motif, neigh_refs_by_species, isolated_lookup_i, tag):
        """Boolean, true iff this atom's real neighbor shell exactly
        matches `motif` (species -> count). Empty motif = "isolated"
        (zero real neighbors of any species)."""
        if not motif:
            distinct_refs = sorted({n["site_index"] for n in ctx.neighbor_cache[isolated_lookup_i]})
            count_expr = sum(any_species_var(r) for r in distinct_refs)
            eq = m.NewBoolVar(f"eq_{tag}_isolated")
            m.Add(count_expr == 0).OnlyEnforceIf(eq)
            m.Add(count_expr != 0).OnlyEnforceIf(eq.Not())
            return eq
        # Closed-world: a plausible neighbor species not named in
        # `motif` must have a count of exactly zero, not "don't care".
        full_motif = {**{sp: 0 for sp in neigh_refs_by_species}, **motif}
        eq_parts = []
        for target_species, n in full_motif.items():
            candidates = neigh_refs_by_species.get(target_species, [])
            count_expr = sum(species_var(r, target_species) for r in candidates)
            eq = m.NewBoolVar(f"eqpart_{tag}_{target_species}{n}")
            m.Add(count_expr == n).OnlyEnforceIf(eq)
            m.Add(count_expr != n).OnlyEnforceIf(eq.Not())
            eq_parts.append(eq)
        return eq_parts[0] if len(eq_parts) == 1 else _reify_and(m, eq_parts, f"eq_{tag}_combined")

    return m, pres, sp_choice, build_eq, presence_literal, label_for


def _iter_motif_candidates(rules_by_species: Dict[str, List[dict]], ctx: _SATContext, build_eq, presence_literal, label_for):
    """For every candidate source atom of every ruled species, yield
    (species, ref, literal, always_true, eqs) where eqs[k] is true iff
    the atom exactly matches its k-th ranked motif. Shared by both rule
    modes (_sat_build_model, _sat_build_model_soft)."""
    for source_species, motifs in rules_by_species.items():
        if not motifs:
            continue
        for src_ref in ctx.source_refs_by_species.get(source_species, []):
            literal, always_true = presence_literal(src_ref, source_species)
            neigh_refs_by_species = _neighbor_refs_by_species(src_ref, ctx)
            eqs = [
                build_eq(mo, neigh_refs_by_species, src_ref, f"{label_for(src_ref)}_{k}")
                for k, mo in enumerate(motifs)
            ]
            yield source_species, src_ref, literal, always_true, eqs


def _sat_build_model(resolved: Dict[str, List[dict]], ctx: _SATContext):
    """HARD rule mode, used when no site group was detected at all (see
    sat_solver): each species' `resolved` motifs are OR'd as a hard
    constraint every candidate atom of that species must satisfy.
    Callers are expected to have already found, per species, the
    smallest ranked prefix of motifs that keeps the model feasible
    (_sat_is_feasible).

    Returns (m, pres, sp_choice)."""
    m, pres, sp_choice, build_eq, presence_literal, label_for = _sat_build_base(ctx)
    for _, src_ref, literal, always_true, eqs in _iter_motif_candidates(resolved, ctx, build_eq, presence_literal, label_for):
        at_least_one = eqs[0] if len(eqs) == 1 else _reify_or(m, eqs, f"anyfav_{label_for(src_ref)}")
        if always_true:
            m.Add(at_least_one == 1)
        else:
            m.Add(at_least_one == 1).OnlyEnforceIf(literal)
    return m, pres, sp_choice


def _sat_is_feasible(resolved, ctx: _SATContext, time_limit: float = 60.0) -> str:
    """Can `resolved`'s HARD constraints be satisfied at all (yes/no,
    not a full solution)? Returns 'feasible', 'infeasible', or 'unknown'
    (CP-SAT hit time_limit without proving either way -- treat as "not
    confirmed feasible", since infeasibility can take much longer to
    prove than a solution takes to find)."""
    m, pres, sp_choice = _sat_build_model(resolved, ctx)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(m)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "feasible"
    if status == cp_model.INFEASIBLE:
        return "infeasible"
    return "unknown"


def _sat_build_model_soft(rules_by_species: Dict[str, List[dict]], ctx: _SATContext):
    """SOFT rule mode, used once a site group was detected (see
    sat_solver): motif rules never become hard constraints, so they can
    never outrank proportional group filling. For each candidate source
    atom and each of its species' ranked motifs, a sat_k boolean is true
    iff that atom is realized as that species AND matches that motif.
    The caller maximizes a rank-weighted sum of these and locks in the
    best achievable score before enumerating solutions.

    Returns (m, pres, sp_choice, rule_terms, rank_satisfied_vars):
    rule_terms is a list of (weight, sat_k) pairs to maximize;
    rank_satisfied_vars[(species, rank)] lists that rank's sat_k vars,
    for reporting how many atoms landed on each rank."""
    m, pres, sp_choice, build_eq, presence_literal, label_for = _sat_build_base(ctx)
    rule_terms: List[Tuple[int, cp_model.IntVar]] = []
    rank_satisfied_vars: Dict[Tuple[str, int], List] = {}
    for source_species, src_ref, literal, always_true, eqs in _iter_motif_candidates(rules_by_species, ctx, build_eq, presence_literal, label_for):
        n = len(eqs)
        for k, eq_k in enumerate(eqs):
            # A motif "match" on a species this atom didn't resolve to
            # earns no credit, so sat_k also requires `literal`.
            sat_k = eq_k if always_true else _reify_and(m, [eq_k, literal], f"sat_{label_for(src_ref)}_{k}")
            rule_terms.append((n - k, sat_k))  # rank 0 (top) worth the most
            rank_satisfied_vars.setdefault((source_species, k), []).append(sat_k)
    return m, pres, sp_choice, rule_terms, rank_satisfied_vars


def _disordered_site_indices(structure: Structure) -> List[int]:
    """Sites with total occupancy < 1 -- a fully-occupied site is always
    real, never an alternate for someone else's slot."""
    return [i for i, site in enumerate(structure) if sum(site.species.as_dict().values()) < 1.0 - 1e-6]


def _pairwise_distances(structure: Structure, indices: Sequence[int], max_dist: float) -> List[Tuple[float, int, int]]:
    """All (distance, i, j) triples among `indices` with distance < max_dist."""
    lattice = structure.lattice
    out = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            d, _ = lattice.get_distance_and_image(structure[i].frac_coords, structure[j].frac_coords)
            if d < max_dist:
                out.append((d, i, j))
    return out


def _species_radius(species: str, species_radii: Optional[Dict[str, float]] = None) -> float:
    """Atomic radius (Å) for `species`, used only by the "radii"
    grouping method. species_radii overrides/supplies a value pymatgen
    lacks; otherwise falls back to Element.atomic_radius, then
    atomic_radius_calculated. Raises rather than guessing if neither
    is available."""
    if species_radii and species in species_radii:
        return float(species_radii[species])
    el = Element(species)
    r = el.atomic_radius if el.atomic_radius is not None else el.atomic_radius_calculated
    if r is None:
        raise ValueError(
            f"No tabulated atomic radius for species {species!r}; pass an "
            f"explicit value via species_radii={{'{species}': <radius in Å>}}."
        )
    return float(r)


def _plot_disorder_group_pies(
    structure: Structure,
    member_indices: Sequence[int],
    title: str,
    out_path: str,
    rotation: str = "15x,15y,0z",
) -> None:
    """Render one site group -- only its own members -- via
    ase.visualize.plot.plot_atoms, with each site's occupancy dict drawn
    as a pie wedge (ASE's atoms.info["occupancy"] + atoms.get_tags())."""
    indices = sorted(set(member_indices))
    # Group members are only guaranteed close under the minimum-image
    # convention -- raw CIF coordinates can sit on opposite sides of the
    # cell despite being physically adjacent, so every member after the
    # first is shifted to its true minimum-image position.
    lattice = structure.lattice
    ref_frac = structure[indices[0]].frac_coords
    symbols, positions, occ_dicts = [], [], {}
    for local_i, orig_i in enumerate(indices):
        site = structure[orig_i]
        symbols.append(_dominant_species(site))
        if local_i == 0:
            positions.append(site.coords)
        else:
            _, jimage = lattice.get_distance_and_image(ref_frac, site.frac_coords)
            positions.append(lattice.get_cartesian_coords(jimage + site.frac_coords))
        occ_dicts[str(local_i)] = site.species.as_dict()

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_tags(list(range(len(indices))))
    atoms.info["occupancy"] = occ_dicts

    max_dist = max(
        np.linalg.norm(np.array(positions[a]) - np.array(positions[b]))
        for a in range(len(positions)) for b in range(a + 1, len(positions))
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#e8e8e8")  # keeps a white vacancy wedge visible
    plot_atoms(atoms, ax, radii=0.315, rotation=rotation)
    ax.set_title(title, fontsize=14)
    ax.text(
        0.5, -0.06, f"Longest pairwise distance: {max_dist:.3f} Å",
        transform=ax.transAxes, ha="center", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), **SAVE_KWARGS)
    plt.close()


def _detect_site_groups_by_radii(
    structure: Structure,
    radius_threshold: float = DEFAULT_RADIUS_THRESHOLD,
    species_radii: Optional[Dict[str, float]] = None,
    search_radius: float = _GROUP_SEARCH_RADIUS,
) -> Tuple[Dict[int, List[int]], Dict[str, float]]:
    """Group disordered (occupancy < 1) sites by an atomic-radii overlap
    test: any pair closer than radius_threshold * (r_i + r_j) -- the
    scaled sum of two sites' dominant-species radii -- is grouped
    transitively, regardless of species. At most one real atom can ever
    come from a group. search_radius just bounds the pairwise-distance
    scan; must exceed the largest cutoff this call can produce.

    A species with an unusually large tabulated radius can chain far
    more sites together than intended -- see _detect_site_groups_by_
    histogram for an alternative independent of tabulated radii.

    Returns (groups, species_radii_used); the latter maps each dominant
    species among the disordered sites to the radius actually applied.
    """
    disordered = _disordered_site_indices(structure)
    dominant_species = {i: _dominant_species(structure[i]) for i in disordered}
    species_radii_used = {sp: _species_radius(sp, species_radii) for sp in set(dominant_species.values())}
    radii = {i: species_radii_used[sp] for i, sp in dominant_species.items()}
    edges = [
        (i, j) for d, i, j in _pairwise_distances(structure, disordered, search_radius)
        if d < radius_threshold * (radii[i] + radii[j])
    ]
    return _union_find_groups(disordered, edges), species_radii_used


def _auto_group_cutoff(
    structure: Structure, indices: Sequence[int],
    search_radius: float = _GROUP_SEARCH_RADIUS, jump_ratio: float = DEFAULT_JUMP_RATIO,
) -> Optional[float]:
    """Pick a group_cutoff (Å) from this structure's own geometry.

    Disorder-split spacings (alternate positions for one physical atom)
    cluster far tighter than any real bond. Collect every pairwise
    distance among disordered sites under search_radius, sort the
    distinct values, and cut at the first gap whose ratio to the
    previous distance reaches jump_ratio -- the boundary right after the
    tight low-distance tier. The FIRST such jump, not the largest gap
    anywhere, since a second short-but-real tier (e.g. cross-bond
    contacts) could otherwise get merged in.

    Returns None only if no pair of disordered sites is within
    search_radius. A single distinct distance is treated as entirely
    disorder-splitting (cutoff just above it). If no gap reaches
    jump_ratio, falls back to the single largest absolute gap.
    """
    dists = _pairwise_distances(structure, indices, search_radius)
    distinct = sorted({round(d, 4) for d, _, _ in dists})
    if not distinct:
        return None
    if len(distinct) == 1:
        return distinct[0] * 1.05
    for lo, hi in zip(distinct, distinct[1:]):
        if hi / lo >= jump_ratio:
            return (lo + hi) / 2
    best_gap, best_cutoff = -1.0, distinct[-1] * 1.05
    for lo, hi in zip(distinct, distinct[1:]):
        if hi - lo > best_gap:
            best_gap, best_cutoff = hi - lo, (lo + hi) / 2
    return best_cutoff


def _detect_site_groups_by_histogram(
    structure: Structure,
    group_cutoff: Optional[float] = None,
    search_radius: float = _GROUP_SEARCH_RADIUS,
    jump_ratio: float = DEFAULT_JUMP_RATIO,
) -> Tuple[Dict[int, List[int]], float]:
    """Group disordered (occupancy < 1) sites: any pair closer than
    group_cutoff (Å) is grouped transitively, regardless of species. At
    most one real atom can ever come from a group.

    group_cutoff=None (default) auto-derives it per structure from the
    pairwise-distance histogram (_auto_group_cutoff), rather than
    assuming a value tuned on a different structure still applies.

    Returns (groups, cutoff_used).
    """
    disordered = _disordered_site_indices(structure)
    if group_cutoff is None:
        group_cutoff = _auto_group_cutoff(structure, disordered, search_radius, jump_ratio) or 0.0
    edges = [(i, j) for _, i, j in _pairwise_distances(structure, disordered, group_cutoff)]
    return _union_find_groups(disordered, edges), group_cutoff


def _group_periodicity_rank(members: Sequence[int], edges: List[Tuple[int, int, np.ndarray]]) -> int:
    """0 if `edges` (each (i, j, jimage): j's periodic image at
    translation jimage lies within the grouping cutoff of i) close up
    within one unit cell -- a genuine, finite disorder slot. 1-3 if
    walking them can reach a site's own periodic image after a net
    lattice translation: the "group" is actually an infinite chain/
    sheet/framework (a site network).

    Standard crystal-net-topology test: grow a spanning tree over the
    edges, giving each site an integer lattice-vector offset relative to
    an arbitrary root. A group that closes up within one cell has every
    non-tree edge consistent with those offsets; an inconsistent edge is
    a cycle that does NOT close within one cell. The rank of those
    "closure" vectors is the network's periodic dimensionality.
    """
    adjacency: Dict[int, List[Tuple[int, np.ndarray]]] = {i: [] for i in members}
    for i, j, jimage in edges:
        adjacency[i].append((j, jimage))
        adjacency[j].append((i, -jimage))

    offset: Dict[int, np.ndarray] = {}
    closure_vectors: List[np.ndarray] = []
    for start in members:
        if start in offset:
            continue
        offset[start] = np.zeros(3)
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor, jimage in adjacency[node]:
                expected = offset[node] + jimage
                if neighbor not in offset:
                    offset[neighbor] = expected
                    stack.append(neighbor)
                elif np.any(expected != offset[neighbor]):
                    closure_vectors.append(expected - offset[neighbor])

    return 0 if not closure_vectors else int(np.linalg.matrix_rank(np.array(closure_vectors)))


def _detect_group_networks(structure: Structure, groups: Dict[int, List[int]], cutoff_of) -> Set[int]:
    """Which of `groups` (gid -> member indices) are periodic self-
    connected networks rather than finite disorder slots, per
    _group_periodicity_rank. cutoff_of(i, j) gives the same closeness
    threshold (Å) that produced `groups` (radii or histogram cutoff).

    For each group, builds a small sub-Structure of just its members and
    calls Structure.get_all_neighbors once to get every periodic image
    of every pair within a generous cutoff, then filters by the exact
    cutoff_of(i, j) (which can vary per species pair). A pair with more
    than one qualifying image at once is the classic sign of a periodic
    self-connection, so all qualifying images are kept, not just the
    nearest."""
    network_gids: Set[int] = set()
    for gid, members in groups.items():
        if len(members) < 2:
            continue
        max_cutoff = max(cutoff_of(members[a], members[b]) for a in range(len(members)) for b in range(a + 1, len(members)))
        sub = Structure(structure.lattice, ["H"] * len(members), [structure[i].frac_coords for i in members])
        local_to_orig = dict(enumerate(members))
        all_neighbors = sub.get_all_neighbors(max_cutoff, include_index=True, include_image=True)

        # get_all_neighbors lists each ordered pair symmetrically (i's
        # neighbor list contains j and vice versa, images negated), so
        # keeping only local_j > local_i already gives each edge once.
        edges = []
        for local_i, neighbor_list in enumerate(all_neighbors):
            for n in neighbor_list:
                local_j = n.index
                if local_j <= local_i:
                    continue
                orig_i, orig_j = local_to_orig[local_i], local_to_orig[local_j]
                if n.nn_distance < cutoff_of(orig_i, orig_j):
                    edges.append((orig_i, orig_j, np.array([int(round(x)) for x in n.image])))

        if edges and _group_periodicity_rank(members, edges) >= 1:
            network_gids.add(gid)
    return network_gids


def _plot_group_cutoff_histogram(
    distances: Sequence[float], cutoff: Optional[float], cutoff_is_auto: bool, out_path: str
) -> None:
    """Histogram of the pairwise distances _auto_group_cutoff scans,
    with the resolved group_cutoff marked."""
    plt.figure(figsize=(8, 5))
    n_bins = min(60, max(10, len(set(round(d, 3) for d in distances))))
    plt.hist(distances, bins=n_bins, color="steelblue", edgecolor="black")
    if cutoff is not None:
        origin = "auto-derived" if cutoff_is_auto else "explicitly passed"
        plt.axvline(cutoff, color="red", linestyle="--", linewidth=2, label=f"group_cutoff = {cutoff:.4f} Å ({origin})")
        plt.legend(fontsize=12)
    plt.xlabel("Pairwise distance between disordered sites (Å)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    _save_fig(out_path)


def _build_bonding_neighbor_cache(
    structure: Structure,
    geometry_species_of: List[str],
    group_of: Dict[int, int],
    groups: Dict[int, List[int]],
    x_diff_weight: float = 0.0,
) -> Dict[int, List[dict]]:
    """Real-bonding-shell neighbors for every site, for checking
    rule/motif satisfaction. Distinct from disorder-group detection: all
    disorder candidates are present at once in `structure`, so a naive
    neighbor search would be swamped by alternate positions of the same
    atom a fraction of an Å apart. For each site i, runs
    CrystalNN(x_diff_weight=x_diff_weight) on a proxy structure excluding
    only i's own group-mates.

    x_diff_weight: how much CrystalNN trusts electronegativity to decide
    what's a bond. 0.0 (default) is pure geometry -- needed when the
    real distinction is same-element at two distances (e.g. ice's
    covalent vs. hydrogen-bonded O-H). Raise toward 1.0 to favor a
    high-electronegativity-difference pair (e.g. Cu-S) over a
    same-element near-contact (e.g. Cu-Cu) that would otherwise be
    miscounted as a bond.
    """
    cnn = CrystalNN(x_diff_weight=x_diff_weight)
    bonding_neighbor_cache: Dict[int, List[dict]] = {}
    for i in range(len(structure)):
        exclude = set(groups[group_of[i]]) - {i} if i in group_of else set()
        keep_indices = [j for j in range(len(structure)) if j not in exclude]
        local_index_of = {orig: local for local, orig in enumerate(keep_indices)}
        proxy = Structure(
            structure.lattice,
            [geometry_species_of[j] for j in keep_indices],
            [structure[j].frac_coords for j in keep_indices],
        )
        neighbors = cnn.get_nn_info(proxy, local_index_of[i])
        bonding_neighbor_cache[i] = [{"site_index": keep_indices[n["site_index"]]} for n in neighbors]
    return bonding_neighbor_cache


def _build_sat_context(
    structure: Structure,
    method: str = "histogram",
    radius_threshold: float = DEFAULT_RADIUS_THRESHOLD,
    species_radii: Optional[Dict[str, float]] = None,
    group_cutoff: Optional[float] = None,
    jump_ratio: float = DEFAULT_JUMP_RATIO,
    x_diff_weight: float = 0.0,
) -> _SATContext:
    """Build the structural/occupancy context the CP-SAT model needs:
    disorder groups, independently disordered sites, each site's real
    bonding-shell neighbors (_build_bonding_neighbor_cache), and the
    global stoichiometric quota each species must hit.

    method picks how disorder groups are detected: "histogram" (default,
    _detect_site_groups_by_histogram, using group_cutoff/jump_ratio) or
    "radii" (_detect_site_groups_by_radii, using
    radius_threshold/species_radii). Group detection works directly on
    pairwise distances; ctx.neighbor_cache (for rule/motif satisfaction)
    is a separate CrystalNN pass over real bonding partners.

    A detected group connected to its own periodic image (a network, not
    a finite slot -- see _detect_group_networks) is flagged in
    network_groups; sat_solver skips that group's usual exclusivity,
    applying only stoichiometry and motif rules to its members.

    Pulled out of sat_solver() so cluster_infeasible_rules() reuses the
    same group definition sat_solver()'s constraints are built from.
    """
    species_radii_used = None
    resolved_group_cutoff = None
    if method == "radii":
        groups, species_radii_used = _detect_site_groups_by_radii(structure, radius_threshold, species_radii)

        def cutoff_of(i, j, _radii=species_radii_used):
            return radius_threshold * (_radii[_dominant_species(structure[i])] + _radii[_dominant_species(structure[j])])
    elif method == "histogram":
        groups, resolved_group_cutoff = _detect_site_groups_by_histogram(structure, group_cutoff, jump_ratio=jump_ratio)

        def cutoff_of(i, j, _cutoff=resolved_group_cutoff):
            return _cutoff
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'radii' or 'histogram'")

    group_of: Dict[int, int] = {i: gid for gid, members in groups.items() for i in members}
    network_groups = _detect_group_networks(structure, groups, cutoff_of)

    # A group's members' occupancies, summed, say whether the slot is
    # ALWAYS occupied (total ~1) or only SOMETIMES (real vacancy
    # disorder) -- only the first needs an "exactly one" floor. The
    # tolerance (not round-to-nearest) avoids a total like 0.9 being
    # wrongly treated as "always occupied".
    group_mandatory: Set[int] = {
        gid for gid, members in groups.items()
        if sum(occ for i in members for occ in structure[i].species.as_dict().values()) >= 1.0 - 1e-3
    }

    labels = [
        site.properties.get("_atom_site_label", f"{site.species_string}{i + 1}")
        for i, site in enumerate(structure)
    ]
    # Majority-vote species per site, only to give the bonding-neighbor
    # CrystalNN pass a single geometry to work with.
    geometry_species_of = [_dominant_species(site) for site in structure]

    # species_of: always-present, single-species, non-grouped sites --
    # no CP-SAT variable needed. member_species: every togglable site's
    # own candidate species, tracked per site (not per group) so later
    # steps can tell which specific member of a group is real.
    species_of: Dict[int, str] = {}
    member_species: Dict[int, List[str]] = {}
    for i, site in enumerate(structure):
        occ_dict = site.species.as_dict()
        if i not in group_of and sum(occ_dict.values()) >= 1.0 - 1e-9:
            species_of[i] = _dominant_species(site)
        else:
            member_species[i] = sorted(occ_dict.keys())

    togglable: Set[int] = set(member_species)

    species_raw_occ_total: Dict[str, float] = {}
    for i in togglable:
        for sp, occ in structure[i].species.as_dict().items():
            species_raw_occ_total[sp] = species_raw_occ_total.get(sp, 0.0) + occ
    stoichiometry = {sp: round(total) for sp, total in species_raw_occ_total.items()}

    source_refs_by_species: Dict[str, List[int]] = {}
    for i, candidates in member_species.items():
        for sp in candidates:
            source_refs_by_species.setdefault(sp, []).append(i)
    for i, sp in species_of.items():
        source_refs_by_species.setdefault(sp, []).append(i)

    bonding_neighbor_cache = _build_bonding_neighbor_cache(structure, geometry_species_of, group_of, groups, x_diff_weight)

    return _SATContext(
        structure=structure,
        labels=labels,
        species_of=species_of,
        member_species=member_species,
        togglable=togglable,
        groups=groups,
        group_of=group_of,
        group_mandatory=group_mandatory,
        network_groups=network_groups,
        source_refs_by_species=source_refs_by_species,
        stoichiometry=stoichiometry,
        neighbor_cache=bonding_neighbor_cache,
        method=method,
        radius_threshold=radius_threshold if method == "radii" else None,
        species_radii_used=species_radii_used,
        group_cutoff=resolved_group_cutoff,
    )


def _neighbor_refs_by_species(src_lookup_i: int, ctx: _SATContext) -> Dict[str, List[int]]:
    """Real neighbors cached at ctx.neighbor_cache[src_lookup_i], grouped
    by every species each could turn out to be. Shared by both rule
    modes and cluster_infeasible_rules, so all agree on what "a neighbor
    of species X" means."""
    neigh_refs_by_species: Dict[str, List[int]] = {}
    for n in ctx.neighbor_cache[src_lookup_i]:
        r = n["site_index"]
        candidate_species = ctx.member_species.get(r) or ([ctx.species_of[r]] if r in ctx.species_of else [])
        for sp in candidate_species:
            neigh_refs_by_species.setdefault(sp, []).append(r)
    return neigh_refs_by_species


def _max_simultaneous_by_group(refs: List[int], ctx: _SATContext) -> int:
    """Most candidate refs (all supplying the same target species) that
    could ever be simultaneously real: sites sharing a disorder group
    contribute at most one real atom between them; a non-grouped site,
    or one in a network group (exempt from exclusivity), is independent.
    A fast necessary condition, not a full feasibility oracle (that's
    CP-SAT itself), and not joint across different target species."""
    distinct_groups: Set[int] = set()
    independent = 0
    for ref in refs:
        gid = ctx.group_of.get(ref)
        if gid is not None and gid not in ctx.network_groups:
            distinct_groups.add(gid)
        else:
            independent += 1
    return independent + len(distinct_groups)


def cluster_infeasible_rules(
    rule_strings: Sequence[str],
    disordered_supercell_file: str,
    method: str = "histogram",
    radius_threshold: float = DEFAULT_RADIUS_THRESHOLD,
    species_radii: Optional[Dict[str, float]] = None,
    group_cutoff: Optional[float] = None,
    jump_ratio: float = DEFAULT_JUMP_RATIO,
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split candidate 'Source-Motif' rules into (kept, dropped,
    unmodeled) from disorder-group geometry alone -- no CP-SAT solve, no
    stoichiometry. Catches one specific case: a motif whose required
    neighbor count for some target species could only ever be met by
    getting 2+ real atoms out of one disorder group at once, which is
    never possible under group exclusivity (a network group is exempt
    from that exclusivity, so is treated as independent sites here too).

    A rule is dropped only if this holds for every candidate source atom
    of its species -- one achievable atom is enough to keep it (sat_
    solver's own OR resolution handles a motif that only fires for some
    atoms). `unmodeled` is separate: a motif can name a species that's
    not a candidate anywhere in this supercell, unrelated to geometry.

    method, radius_threshold, species_radii, group_cutoff, jump_ratio are
    passed through to _build_sat_context.

    Returns (kept_rules, dropped, unmodeled); `dropped` and `unmodeled`
    are each a list of (rule_string, reason) pairs.
    """
    structure = Structure.from_file(disordered_supercell_file)
    ctx = _build_sat_context(structure, method, radius_threshold, species_radii, group_cutoff, jump_ratio)
    species_present: Set[str] = set(ctx.species_of.values()) | {
        sp for candidates in ctx.member_species.values() for sp in candidates
    }

    parsed = _parse_candidates(rule_strings)
    kept: List[str] = []
    dropped: List[Tuple[str, str]] = []
    unmodeled: List[Tuple[str, str]] = []

    for raw, (source_species, motif, _annotation) in zip(rule_strings, parsed):
        if not motif:
            kept.append(raw)  # "isolated" makes no neighbor-count claim, nothing to check
            continue

        missing = sorted((set(motif) | {source_species}) - species_present)
        if missing:
            role = "source" if source_species in missing else "target"
            unmodeled.append((
                raw,
                f"species {', '.join(missing)} ({role}) not present as a candidate species "
                f"in this supercell, so this rule is inapplicable here",
            ))
            continue

        feasible_anywhere = False
        best_reason = None
        for src_ref in ctx.source_refs_by_species.get(source_species, []):
            neigh_refs_by_species = _neighbor_refs_by_species(src_ref, ctx)
            atom_ok = True
            for target_species, n_needed in motif.items():
                achievable = _max_simultaneous_by_group(neigh_refs_by_species.get(target_species, []), ctx)
                if achievable < n_needed:
                    atom_ok = False
                    if best_reason is None:
                        best_reason = (
                            f"needs {n_needed} simultaneous {target_species} neighbor(s), "
                            f"but at most {achievable} could ever be real at once"
                        )
            if atom_ok:
                feasible_anywhere = True
                break

        if feasible_anywhere:
            kept.append(raw)
        else:
            dropped.append((raw, best_reason or "infeasible for every candidate source atom"))

    return kept, dropped, unmodeled


def _trim_by_correlation(rule_strings: Sequence[str], motif_correlations: Dict[str, float]) -> Tuple[List[str], List[Tuple[str, float]]]:
    """Drop soft-mode rules whose SHAP correlation (shap_motifs'
    "Correlation" column: signed by which way the motif pushes the
    target property) is positive -- assumed undesirable, e.g. less
    stable if the target is formation energy. A rule with no entry in
    motif_correlations is kept: no evidence either way. Keys must match
    `rule_strings` exactly (the same "Source-Motif" strings sat_solver
    takes).

    Returns (kept, trimmed); trimmed pairs each dropped rule with its
    correlation value."""
    kept: List[str] = []
    trimmed: List[Tuple[str, float]] = []
    for raw in rule_strings:
        corr = motif_correlations.get(raw)
        if corr is not None and corr > 0:
            trimmed.append((raw, corr))
        else:
            kept.append(raw)
    return kept, trimmed


# ==============================================================
# User Functions
# ==============================================================

def shap_motifs(
    df_motifs: str,
    df_energies: str,
    out_dir: str = ".",
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
) -> Sequence[str]:
    """Rank motifs by how strongly they drive a target property.

    Trains a random forest to predict df_energies from df_motifs' motif
    counts, then uses SHAP to explain it: each motif gets an importance
    (how much it matters) and a signed correlation (which direction it
    pushes the target). Writes plots and a CSV of the ranking to out_dir.

    Args:
        test_size: Fraction of structures held out for the test split.
        random_state: Seed for the split and the forest. None draws a
            fresh seed each call (printed either way, so a "random" run
            can be reproduced).

    Returns the motif names sorted by that signed correlation.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)
    seed = _resolve_seed(random_state)

    df = _merge_df(df_motifs, df_energies)
    X, y = _prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    model = _train_model(X_train, y_train, random_state=seed)
    _report_performance(model, X_test, y_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    _plot_shap_summary(shap_values, X, out_dir / "SHAP_summary.png")
    _plot_shap_bar(shap_values, X, out_dir / "SHAP_MeanAbsShap.png")

    importance = _build_importance_table(X, shap_values)
    importance.to_csv(out_dir / "SHAP_clique_importance.csv", index=False)
    print("\nTop 20 most influential cliques:")
    print(importance.head(20), "\n")
    _plot_correlation_bar(importance, out_dir / "SHAP_Correlation.png")

    return importance.sort_values("Correlation")["Motif"].tolist()


def _summarize_groups(ctx: _SATContext):
    """Bucket group ids by (size, species set, mandatory, is_network).
    Returns (counts, representative_gid): counts[sig] = how many groups
    share it, representative_gid[sig] = one example gid -- used for both
    the printed summary and the one-picture-per-signature diagnostics."""
    counts: Counter = Counter()
    representative: Dict[tuple, int] = {}
    for gid, members in ctx.groups.items():
        species = tuple(sorted({sp for i in members for sp in ctx.member_species[i]}))
        sig = (len(members), species, gid in ctx.group_mandatory, gid in ctx.network_groups)
        counts[sig] += 1
        representative.setdefault(sig, gid)
    return counts, representative


def _print_group_summary(ctx: _SATContext, out_dir: str, structure: Structure, group_cutoff_arg, plot_diagnostics: bool) -> None:
    """Print the site-group summary (and, if requested, write one
    diagnostic picture per distinct group signature) for sat_solver."""
    if ctx.method == "radii":
        radii_str = ", ".join(f"{sp}={r:.3f} A" for sp, r in sorted(ctx.species_radii_used.items()))
        print(f"Site groups: radius_threshold={ctx.radius_threshold:g} x (r_i + r_j), radii {radii_str}")
    else:
        print(f"Site groups: histogram group cutoff at {ctx.group_cutoff:.4f} A")

    if not ctx.groups:
        print("  none")
        return

    counts, representative = _summarize_groups(ctx)
    for (size, species, mandatory, is_network), n_groups in sorted(counts.items()):
        kind = "network" if is_network else ("occ==1" if mandatory else "occ<=1")
        print(f"  {n_groups}x size-{size} ({'/'.join(species)}, {kind})")

    if ctx.network_groups:
        n_net = len(ctx.network_groups)
        print(f"  -> {n_net} group{'s' if n_net != 1 else ''} classified as network")

    if not plot_diagnostics:
        return
    group_plot_dir = os.path.join(out_dir, "_positional_clusters")
    if os.path.isdir(group_plot_dir):
        shutil.rmtree(group_plot_dir)  # never leave a stale picture from a previous run
    os.makedirs(group_plot_dir, exist_ok=True)

    if ctx.method == "histogram":
        disordered = _disordered_site_indices(structure)
        dists = [d for d, _, _ in _pairwise_distances(structure, disordered, _GROUP_SEARCH_RADIUS)]
        _plot_group_cutoff_histogram(
            dists, ctx.group_cutoff, group_cutoff_arg is None,
            os.path.join(group_plot_dir, "group_cutoff_histogram.png"),
        )
    for (size, species, mandatory, is_network), gid in representative.items():
        kind = "network" if is_network else ("mandatory" if mandatory else "optional")
        title = (
            f"{counts[(size, species, mandatory, is_network)]}x size-{size} {'/'.join(species)} group ({kind}) "
            f"[representative: {ctx.labels[gid]} et al.]"
        )
        _plot_disorder_group_pies(
            structure, ctx.groups[gid], title,
            os.path.join(group_plot_dir, f"group_size{size}_{'-'.join(species)}_{kind}.png"),
        )
    print(f"  -> {len(representative)} group image(s) written to {group_plot_dir}/")


def _group_rules_by_species(
    rules: Sequence[str],
) -> Tuple[List[str], Dict[str, List[dict]], Dict[str, List[Optional[str]]]]:
    """Parse 'Source-Motif' rule strings and bucket them by source
    species, ranked in the order each species first appears in `rules`.
    Returns (species_order, rules_by_species, annotations_by_species) --
    the last is index-aligned with rules_by_species[sp], carrying each
    rule's "(...)" local-symmetry tag (or None) purely for display, so
    two annotated variants of the same neighbor-count motif still print
    as visibly distinct rules even though they solve identically."""
    species_order: List[str] = []
    rules_by_species: Dict[str, List[dict]] = {}
    annotations_by_species: Dict[str, List[Optional[str]]] = {}
    for source_species, motif, annotation in _parse_candidates(rules):
        if source_species not in rules_by_species:
            rules_by_species[source_species] = []
            annotations_by_species[source_species] = []
            species_order.append(source_species)
        rules_by_species[source_species].append(motif)
        annotations_by_species[source_species].append(annotation)
    return species_order, rules_by_species, annotations_by_species


def _resolve_hard_rules(
    rules_by_species: Dict[str, List[dict]],
    species_order: List[str],
    ctx: _SATContext,
    time_limit: float,
    annotations_by_species: Optional[Dict[str, List[Optional[str]]]] = None,
) -> Dict[str, List[dict]]:
    """For each species, binary-search the smallest ranked prefix of its
    motifs that keeps the whole model feasible (feasibility is monotonic
    in prefix length, since a longer prefix only adds OR disjuncts), and
    OR that prefix in. 'unknown' (a feasibility check timing out) is
    treated conservatively as not-yet-feasible. Prints each species'
    resolution, tagging each printed motif with its "(...)" annotation
    (if any, from annotations_by_species) so two annotated variants of
    the same neighbor-count motif stay visually distinct. Returns the
    resolved {species: accepted motif prefix}."""
    print("\nResolving rules (hard -- no site groups detected):")
    resolved: Dict[str, List[dict]] = {}
    for source_species in species_order:
        motifs = rules_by_species[source_species]
        n_motifs = len(motifs)
        uncertain = False

        def check(k):
            trial = dict(resolved)
            trial[source_species] = motifs[:k]
            return _sat_is_feasible(trial, ctx, time_limit=time_limit)

        status_n = check(n_motifs)
        uncertain |= status_n == "unknown"
        if status_n != "feasible":
            accepted_k = None
        else:
            lo, hi = 1, n_motifs
            while lo < hi:
                mid = (lo + hi) // 2
                status_mid = check(mid)
                uncertain |= status_mid == "unknown"
                hi, lo = (mid, lo) if status_mid == "feasible" else (hi, mid + 1)
            accepted_k = lo

        if accepted_k is None:
            note = " (unconfirmed: some check timed out)" if uncertain else ""
            print(f"  {source_species}: unsatisfiable with any candidate motif -> unconstrained{note}")
        else:
            resolved[source_species] = motifs[:accepted_k]
            anns = annotations_by_species[source_species] if annotations_by_species else [None] * n_motifs
            granted_str = " OR ".join(
                ("isolated" if not mo else ", ".join(f"{sp}{n}" for sp, n in mo.items()))
                + (f" {ann}" if ann else "")
                for mo, ann in zip(motifs[:accepted_k], anns[:accepted_k])
            )
            note = " (upper bound: some check timed out unconfirmed)" if uncertain else ""
            print(f"  {source_species}: accepted [{granted_str}]{note}")
    return resolved


def _resolve_soft_rules(
    rules_by_species: Dict[str, List[dict]],
    species_order: List[str],
    ctx: _SATContext,
    time_limit: float,
    annotations_by_species: Optional[Dict[str, List[Optional[str]]]] = None,
):
    """Build the soft-rule model, maximize its rank-weighted motif score,
    print the result, then lock that score in as a hard floor (with a
    solution hint, since re-satisfying an exact equality on a large
    weighted sum can otherwise be slow for CP-SAT to rediscover from
    scratch). Returns (model, pres, sp_choice, all_toggle_vars,
    rule_terms) ready for sat_solver's enumeration loop, or None if no
    structure satisfies the hard constraints at all."""
    print("\nMotif rules (soft):")
    for sp in species_order:
        motifs = rules_by_species[sp]
        anns = annotations_by_species[sp] if annotations_by_species else [None] * len(motifs)
        desc = " > ".join(
            ("Ø" if not mo else "".join(f"{s}{n if n != 1 else ''}" for s, n in mo.items()))
            + (f" {ann}" if ann else "")
            for mo, ann in zip(motifs, anns)
        )
        print(f"  {sp}-: {desc}")

    model, pres, sp_choice, rule_terms, rank_satisfied_vars = _sat_build_model_soft(rules_by_species, ctx)
    all_toggle_vars = list(pres.values()) + list(sp_choice.values())
    if not rule_terms:
        return model, pres, sp_choice, all_toggle_vars, rule_terms

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.max_time_in_seconds = time_limit
    objective_expr = sum(weight * var for weight, var in rule_terms)
    model.Maximize(objective_expr)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"\nstatus: {solver.StatusName(status)}")
        print("No structure satisfies the hard site-group/stoichiometry constraints at all.")
        return None

    best_score = round(solver.ObjectiveValue())
    print(f"\nBest achievable motif-rule score: {best_score} (rank-weighted; ties broken freely)")

    model.Add(objective_expr == best_score)
    model.ClearObjective()
    for v in all_toggle_vars:
        model.AddHint(v, solver.Value(v))
    return model, pres, sp_choice, all_toggle_vars, rule_terms


def _enumerate_solutions(
    model, pres, sp_choice, all_toggle_vars, rule_terms, ctx: _SATContext, structure: Structure,
    out_dir: str, max_solutions: int, randomize_solutions: bool, random_state: Optional[int],
) -> None:
    """Enumerate distinct solutions (solve -> write CIF -> block -> re-
    solve) up to max_solutions, printing a final status line. If
    randomize_solutions, each solve minimizes a fresh random objective
    over every toggle variable first, so solutions are drawn from
    different parts of the feasible space rather than each being the
    nearest tweak of the last (the blocking constraints below still
    apply regardless, so exhaustiveness detection is unaffected)."""
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.max_time_in_seconds = 60

    solutions_dir = os.path.join(out_dir, "_sat_solutions")
    if os.path.isdir(solutions_dir):
        shutil.rmtree(solutions_dir)
    os.makedirs(solutions_dir, exist_ok=True)

    rng = random.Random(_resolve_seed(random_state)) if randomize_solutions else None

    n_found = 0
    final_status = None
    while n_found < max_solutions:
        if rng is not None:
            model.Minimize(sum(rng.randint(-1000, 1000) * v for v in all_toggle_vars))
        final_status = status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        n_found += 1

        # Fixed sites always keep their coordinate; togglable sites keep
        # whichever candidate position was actually selected.
        species_out, coords_out = [], []
        for i in range(len(structure)):
            if i in ctx.species_of:
                species_out.append(ctx.species_of[i])
                coords_out.append(structure[i].frac_coords)
            elif solver.Value(pres[i]):
                candidates = ctx.member_species[i]
                sp = candidates[0] if len(candidates) == 1 else next(s for s in candidates if solver.Value(sp_choice[(i, s)]))
                species_out.append(sp)
                coords_out.append(structure[i].frac_coords)

        Structure(structure.lattice, species_out, coords_out).to(filename=os.path.join(solutions_dir, f"solution_{n_found}.cif"))

        if rule_terms:
            # Re-seed the hint from the solution just found -- it's about
            # to be blocked, but a near neighbor of it still helps CP-SAT
            # re-satisfy the locked rule-score equality quickly.
            model.ClearHints()
            for v in all_toggle_vars:
                model.AddHint(v, solver.Value(v))

        model.Add(sum((1 - v) if solver.Value(v) else v for v in all_toggle_vars) >= 1)

    if n_found == 0:
        print(f"\nstatus: {solver.StatusName(final_status)}")
        print("No solution found - the resolved rules aren't jointly satisfiable.")
    elif final_status == cp_model.INFEASIBLE:
        print("\nstatus: exhaustive")
        print(f"Total solutions written: {n_found}")
    elif n_found >= max_solutions:
        print("\nstatus: stopped - max_solutions cap reached")
        print(f"Total solutions written: {n_found} (capped)")
    else:
        print("\nstatus: stopped - a solve attempt timed out")
        print(f"Total solutions written: {n_found} (inconclusive)")


def sat_solver(disordered_supercell_file, rules,
               method: str = "histogram",
               radius_threshold: float = DEFAULT_RADIUS_THRESHOLD,
               species_radii: Optional[Dict[str, float]] = None,
               group_cutoff: Optional[float] = None,
               jump_ratio: float = DEFAULT_JUMP_RATIO,
               x_diff_weight: float = 0.0,
               max_solutions: int = 400,
               out_dir: str = ".",
               feasibility_time_limit: float = 60.0,
               plot_diagnostics: bool = True,
               randomize_solutions: bool = True,
               random_state: Optional[int] = None,
               motif_correlations: Optional[Dict[str, float]] = None,
               p5: bool = True):
    """Resolve a disordered structure into fully-ordered candidates via
    CP-SAT, ranked against 'Source-Motif' rules, then enumerate every
    distinct solution up to a cap.

    Rule handling:
    - p5=False: motif rules are switched off. `rules` is never even
      parsed -- only site-group exclusivity and stoichiometry (global
      and per-site-type) decide the outcome.
    - p5=True, no site group (every disordered site independent): rules
      are HARD -- the smallest ranked motif prefix that keeps the model
      feasible becomes a required constraint per species
      (_resolve_hard_rules).
    - p5=True, any site group (finite or a network -- see
      _detect_group_networks): group/stoichiometry constraints stay
      HARD; motifs become SOFT suggestions layered on top, maximized
      then locked in before enumerating (_resolve_soft_rules).

    Every motif is closed-world: a species missing from it must have
    zero neighbors ("Fe-S4" = exactly 4 S, nothing else).

    method: how disorder groups are detected -- "histogram" (default,
    group_cutoff/jump_ratio) or "radii" (radius_threshold/species_radii).

    x_diff_weight: passed to CrystalNN for the real bonding-shell
    neighbor cache rules are checked against (_build_bonding_neighbor_cache).

    feasibility_time_limit: per-solve budget for hard-mode prefix checks
    or the soft-mode maximize step. Enumeration always gets 60s/solution.

    plot_diagnostics: writes one picture per disorder-group signature to
    out_dir/_positional_clusters/. Solution CIFs always go to
    out_dir/_sat_solutions/.

    randomize_solutions: draw solutions from different parts of the
    feasible space instead of each nearest to the last. random_state
    seeds the draw (always printed, so it can be repeated).

    motif_correlations: optional {rule_string: correlation} map (e.g.
    from shap_motifs) -- in soft mode, drops any rule with a positive
    correlation before solving (_trim_by_correlation). No effect in
    hard mode or when p5=False.
    """
    structure = Structure.from_file(disordered_supercell_file)
    os.makedirs(out_dir, exist_ok=True)
    ctx = _build_sat_context(structure, method, radius_threshold, species_radii, group_cutoff, jump_ratio, x_diff_weight)

    print("SAT Solver\n----------")
    _print_group_summary(ctx, out_dir, structure, group_cutoff, plot_diagnostics)

    print("Site assignment:")
    if not ctx.stoichiometry:
        print("  none")
    else:
        for sp, target in ctx.stoichiometry.items():
            print(f"  {sp}: {target} of {len(ctx.source_refs_by_species.get(sp, []))} candidate sites")

    if not p5:
        # Motif rules off: keep only site-group/stoichiometry structure.
        print("\nMotif rules disabled (p5=False)")
        model, pres, sp_choice, _, _, _ = _sat_build_base(ctx)
        all_toggle_vars = list(pres.values()) + list(sp_choice.values())
        rule_terms = []
    elif not ctx.groups:
        # No site group, no network: rules are HARD.
        species_order, rules_by_species, annotations_by_species = _group_rules_by_species(rules)
        resolved = _resolve_hard_rules(rules_by_species, species_order, ctx, feasibility_time_limit, annotations_by_species)
        model, pres, sp_choice = _sat_build_model(resolved, ctx)
        all_toggle_vars = list(pres.values()) + list(sp_choice.values())
        rule_terms = []
    else:
        # A site group or a site network: rules are SOFT.
        if motif_correlations:
            rules, trimmed = _trim_by_correlation(rules, motif_correlations)
            if trimmed:
                print(f"\nDropped {len(trimmed)} soft rule(s) with positive SHAP correlation:")
                for raw, corr in trimmed:
                    print(f"  {raw} (correlation={corr:+.4f})")
        species_order, rules_by_species, annotations_by_species = _group_rules_by_species(rules)
        resolved_soft = _resolve_soft_rules(rules_by_species, species_order, ctx, feasibility_time_limit, annotations_by_species)
        if resolved_soft is None:
            return
        model, pres, sp_choice, all_toggle_vars, rule_terms = resolved_soft

    _enumerate_solutions(
        model, pres, sp_choice, all_toggle_vars, rule_terms, ctx, structure,
        out_dir, max_solutions, randomize_solutions, random_state,
    )