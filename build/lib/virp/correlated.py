"""
Tools for disordered crystal structures (sites shared by >1 species/
vacancy, occupancies summing to 1).

shap_motifs(): ranks "motifs" (named local coordination environments,
e.g. "Cu-S4" = Cu with 4 S neighbors) by SHAP correlation with a target
property (e.g. formation energy).

sat_solver() / cluster_infeasible_rules(): given a disordered structure
and ranked motif rules, use CP-SAT to pick one real occupant per
disordered site (preserving overall stoichiometry) so the result
satisfies those rules. sat_solver enumerates every distinct solution,
up to a cap.

Disorder groups: disordered sites close enough together are the same
physical slot (at most one can be real). `method` picks the closeness
test (see _build_sat_context): "histogram" (default) auto-derives a
cutoff from this structure's own distance histogram; "radii" groups
sites closer than radius_threshold * (r_i + r_j) -- simpler, but a
species with a large tabulated radius can over-merge groups.
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

# ASE's default H color is white -- invisible against the page and the
# white "vacancy" wedge. Recolor it once, globally, so it's visible.
_ase_io_utils.default_colors[_ase_atomic_numbers["H"]] = np.array([0.75, 0.75, 0.75])


SAVE_KWARGS = dict(dpi=300, bbox_inches="tight")

# Outer bound (Å) for the pairwise-distance scan. Must exceed the largest
# radius_threshold * (r_i + r_j) you expect to apply.
_GROUP_SEARCH_RADIUS = 4.0

# "radii" method: group sites closer than this * (r_i + r_j). 1.0 = group
# when their atomic radii would literally overlap.
DEFAULT_RADIUS_THRESHOLD = 1.0

# "histogram" method: the disorder-split/real-bond boundary is the first
# distance gap whose ratio to the previous distance reaches this. See
# _auto_group_cutoff.
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

    X = X.loc[:, X.nunique() > 1]          # drop constant columns
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"Structures : {len(df)}")
    print(f"Features   : {X.shape[1]}")
    return X, y


def _resolve_seed(random_state: Optional[int]) -> int:
    """random_state=None draws a fresh seed, an int pins one. Always
    printed, so a "random" run can be repeated later."""
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
    (which way it pushes the target). Correlation isn't a raw Pearson r
    -- noisy on these mostly-zero columns -- it's MeanAbsSHAP signed by
    the Pearson correlation's direction."""
    correlations = _pearson_corr_columns(X.to_numpy(dtype=float), shap_values)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    signed_importance = np.sign(correlations) * mean_abs_shap
    importance = pd.DataFrame(
        {
            "Motif": X.columns,
            "MeanAbsSHAP": mean_abs_shap,
            "Correlation": signed_importance,
        }
    ).sort_values("MeanAbsSHAP", ascending=False)
    return importance


def _plot_correlation_bar(importance: pd.DataFrame, out_path: str) -> None:
    corr_df = importance.sort_values("Correlation")
    colors = np.where(corr_df["Correlation"] > 0, "red", "green")

    plt.figure(figsize=(8, max(3, len(corr_df) * 0.15)))
    bars = plt.barh(corr_df["Motif"], corr_df["Correlation"], color=colors)
    plt.axvline(0, color="black", linewidth=1)

    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        plt.text(
            width / 2, y, f"{width:.2f}",
            ha="center", va="center", color="white", fontsize=12, fontweight="bold",
        )

    plt.xlabel("Signed SHAP Importance (sign = feature–SHAP correlation)", fontsize=14)
    plt.ylabel("Motif", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    _save_fig(out_path)


# ==============================================================
# Ancillary Functions (sat)
# ==============================================================

def _parse_candidates(rule_strings):
    """Parse 'Source-Motif' strings (e.g. "Cu-S4Fe2") into
    (source_species, motif_dict) pairs, e.g. ("Cu", {"S": 4, "Fe": 2}).
    A bare species defaults to count 1; a species absent from the motif
    is treated as count 0 elsewhere (build_eq), not as unconstrained."""
    motif_token_re = re.compile(r"([A-Z][a-z]?)(\d*)")
    parsed = []
    for raw in rule_strings:
        raw = raw.strip()
        if "-" not in raw:
            raise ValueError(f"Malformed rule (expected 'Source-Motif'): {raw!r}")
        source, motif_str = raw.split("-", 1)
        source = source.strip()
        motif_str = motif_str.strip()
        motif = {}
        pos = 0
        for m in motif_token_re.finditer(motif_str):
            species, count_str = m.groups()
            count = int(count_str) if count_str else 1
            motif[species] = motif.get(species, 0) + count
            pos = m.end()
        if pos != len(motif_str):
            raise ValueError(f"Could not fully parse motif {motif_str!r} in rule: {raw!r}")
        parsed.append((source, motif))
    return parsed


def _dominant_species(site) -> str:
    """Majority-occupancy species at one site. Only for a stand-in label
    (non-disordered sites, the neighbor-detection geometry proxy) --
    never to decide a disordered site's real species, which must stay a
    CP-SAT choice (member_species) or a minority occupant could never
    be selected."""
    occ_dict = site.species.as_dict()
    return max(occ_dict, key=occ_dict.get)


@dataclass
class _SATContext:
    """Read-only structural/occupancy data the CP-SAT model builder
    needs, built once and passed around explicitly.

    Every site is either:
    - togglable (in member_species): a disorder-group member or an
      independently disordered site. Gets a presence variable pres[i],
      plus a species-choice variable if it has multiple candidates.
    - fixed (in species_of): occupancy 1, one species, no variable
      needed.
    Grouped sites also appear in group_of/groups, driving the <=1 (or
    ==1, if group_mandatory) exclusivity constraint on a group's
    pres[i] -- separate from which species wins (member_species).
    """
    structure: Structure
    labels: List[str]
    species_of: Dict[int, str]
    member_species: Dict[int, List[str]]
    togglable: Set[int]
    groups: Dict[int, List[int]]
    group_of: Dict[int, int]
    group_mandatory: Set[int]
    source_refs_by_species: Dict[str, List[int]]
    stoichiometry: Dict[str, int]
    neighbor_cache: Dict[int, List[dict]]
    method: str
    radius_threshold: Optional[float]
    species_radii_used: Optional[Dict[str, float]]
    group_cutoff: Optional[float]


# Union-find (disjoint-set), used by both group-detection methods to
# chain sites transitively: A-B and B-C grouped means A, B, C all group.

def _find(x, parent):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(x, y, parent):
    rx, ry = _find(x, parent), _find(y, parent)
    if rx != ry:
        parent[rx] = ry


def _sat_build_model(resolved, ctx: _SATContext):
    """Build a CP-SAT model: structural/group-exclusivity/stoichiometric
    hard constraints, plus an OR-favoured neighbor-count constraint for
    each (species, motif-list) pair in `resolved`.

    pres[i]: boolean, is site i realized (every togglable site gets
    one). sp_choice[i, sp]: boolean species choice, only for a site with
    more than one candidate species. Group exclusivity sums pres[i]
    across a group's members. Occupancy is enforced per species overall
    and per stratum (see the stratum block below).
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
        total = sum(pres[i] for i in members)
        if gid in ctx.group_mandatory:
            # Occupancies sum to ~1: the slot is never really vacant, so
            # exactly one member must be realized, not merely at most one.
            m.Add(total == 1)
        else:
            m.Add(total <= 1)

    def species_var(ref, sp):
        """0/1 expression for 'ref is realized as species sp' -- a CP-SAT
        literal where that's a real choice, else a python constant."""
        if ref in ctx.member_species:
            candidates = ctx.member_species[ref]
            if sp not in candidates:
                return 0
            return sp_choice[(ref, sp)] if len(candidates) > 1 else pres[ref]
        return 1 if ctx.species_of.get(ref) == sp else 0

    # Force the count of sites realized as each species (summed across
    # every site that could possibly be it) to hit its target exactly.
    # Fixed sites contribute a python constant, so are dropped from the
    # sum (already baked into the target).
    for sp, target in ctx.stoichiometry.items():
        terms = [
            species_var(ref, sp)
            for ref in ctx.source_refs_by_species.get(sp, [])
        ]
        terms = [t for t in terms if not isinstance(t, int)]
        if terms:
            m.Add(sum(terms) == target)

    # Occupancy-proportional sub-stoichiometry: sites sharing the exact
    # same occupancy signature (e.g. every {"Fe":0.083,"Cu":0.25} site)
    # are repeated instances of one physical site type. The coarse
    # species-wide target above lets CP-SAT concentrate occupancy on one
    # such type and starve another while still matching overall --
    # physically implausible. Pin each type's own share too, whenever
    # every type's rounded share still sums to the coarse target
    # (skipped otherwise, to avoid a rounding edge case forcing
    # infeasibility).
    strata: Dict[tuple, List[int]] = {}
    for i in ctx.togglable:
        sig = tuple(sorted((sp, round(occ, 6)) for sp, occ in ctx.structure[i].species.as_dict().items()))
        strata.setdefault(sig, []).append(i)

    stratum_targets_by_species: Dict[str, List[Tuple[List[int], int]]] = {}
    for sig, members in strata.items():
        for sp, occ in dict(sig).items():
            target = round(occ * len(members))
            stratum_targets_by_species.setdefault(sp, []).append((members, target))

    for sp, stratum_targets in stratum_targets_by_species.items():
        if sum(target for _, target in stratum_targets) != ctx.stoichiometry.get(sp, 0):
            continue
        for members, target in stratum_targets:
            terms = [species_var(ref, sp) for ref in members]
            terms = [t for t in terms if not isinstance(t, int)]
            if terms:
                m.Add(sum(terms) == target)

    def any_species_var(ref):
        """0/1 expression for 'ref is realized as SOME species'."""
        return pres[ref] if ref in pres else 1

    def presence_literal(ref, sp):
        """(literal, always_true) for a constraint conditional on 'ref is
        species sp'. always_true means ref is an always-present fixed
        site of exactly that species, so the constraint is unconditional.

        Must be the SPECIES CHOICE, not mere presence, at a site with
        multiple candidates: pres[ref] is true for EITHER candidate, so
        using it would wrongly force sp's motif onto a site that
        resolved to a different species."""
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
        # `motif` must have a count of exactly zero, not "don't care" --
        # else {"O": 1} would allow extra neighbors of any other species.
        implied_zero_species = set(neigh_refs_by_species) - set(motif)
        full_motif = dict(motif)
        for sp in implied_zero_species:
            full_motif[sp] = 0
        eq_parts = []
        for target_species, n in full_motif.items():
            candidates = neigh_refs_by_species.get(target_species, [])
            count_expr = sum(species_var(r, target_species) for r in candidates)
            eq = m.NewBoolVar(f"eqpart_{tag}_{target_species}{n}")
            m.Add(count_expr == n).OnlyEnforceIf(eq)
            m.Add(count_expr != n).OnlyEnforceIf(eq.Not())
            eq_parts.append(eq)
        if len(eq_parts) == 1:
            return eq_parts[0]
        combined = m.NewBoolVar(f"eq_{tag}_combined")
        m.AddBoolAnd(eq_parts).OnlyEnforceIf(combined)
        m.AddBoolOr([e.Not() for e in eq_parts]).OnlyEnforceIf(combined.Not())
        return combined

    for source_species, motifs in resolved.items():
        if not motifs:
            continue
        source_refs = ctx.source_refs_by_species.get(source_species, [])
        for src_ref in source_refs:
            literal, always_true = presence_literal(src_ref, source_species)
            neigh_refs_by_species = _neighbor_refs_by_species(src_ref, ctx)

            eq_favoured = [
                build_eq(mo, neigh_refs_by_species, src_ref, f"{label_for(src_ref)}_{k}")
                for k, mo in enumerate(motifs)
            ]
            at_least_one = eq_favoured[0] if len(eq_favoured) == 1 else None
            if at_least_one is None:
                at_least_one = m.NewBoolVar(f"anyfav_{label_for(src_ref)}")
                m.AddBoolOr(eq_favoured).OnlyEnforceIf(at_least_one)
                m.AddBoolAnd([e.Not() for e in eq_favoured]).OnlyEnforceIf(at_least_one.Not())
            if always_true:
                m.Add(at_least_one == 1)
            else:
                m.Add(at_least_one == 1).OnlyEnforceIf(literal)
    return m, pres, sp_choice


def _sat_is_feasible(resolved, ctx: _SATContext, time_limit: float = 60.0) -> str:
    """Can `resolved`'s constraints be satisfied at all (not asking for
    a full solution, just yes/no)? Returns 'feasible', 'infeasible', or
    'unknown' (CP-SAT hit time_limit without proving either way -- treat
    as "not confirmed feasible", not False, since infeasibility can take
    much longer to prove than a solution takes to find)."""
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


def _disordered_site_indices(structure: Structure) -> List[int]:
    """Sites with total occupancy < 1. A fully-occupied site is always
    real, never an alternate for someone else's slot, so it's excluded
    from group detection regardless of proximity."""
    return [
        i for i, site in enumerate(structure)
        if sum(site.species.as_dict().values()) < 1.0 - 1e-6
    ]


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
    atomic_radius_calculated. Raises rather than guessing when neither
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
    """Render one site group -- only its own members, nothing else --
    via ase.visualize.plot.plot_atoms. Each site's occupancy dict is
    passed to ASE as atoms.info["occupancy"] + atoms.get_tags(), its
    native mechanism for a pie wedge per candidate species."""
    indices = sorted(set(member_indices))
    # A group's members are only guaranteed close under the minimum-image
    # convention -- raw CIF coordinates can sit on opposite sides of the
    # unit cell despite being physically close, so every member after the
    # first is shifted to its true minimum-image position.
    lattice = structure.lattice
    ref_frac = structure[indices[0]].frac_coords
    symbols, positions, occ_dicts = [], [], {}
    for local_i, orig_i in enumerate(indices):
        site = structure[orig_i]
        occ_dict = site.species.as_dict()
        symbols.append(_dominant_species(site))
        if local_i == 0:
            positions.append(site.coords)
        else:
            _, jimage = lattice.get_distance_and_image(ref_frac, site.frac_coords)
            positions.append(lattice.get_cartesian_coords(jimage + site.frac_coords))
        occ_dicts[str(local_i)] = occ_dict

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_tags(list(range(len(indices))))
    atoms.info["occupancy"] = occ_dicts

    max_dist = max(
        np.linalg.norm(np.array(positions[a]) - np.array(positions[b]))
        for a in range(len(positions)) for b in range(a + 1, len(positions))
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    # Background is light grey, not white, so the white "vacancy" wedge
    # ASE draws for a partially-occupied site stays visible against it.
    fig.patch.set_facecolor("#e8e8e8")
    plot_atoms(atoms, ax, radii=0.315, rotation=rotation)  # 0.45 * 0.7
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
    scaled sum of two sites' own dominant-species radii -- is unioned
    transitively into one group, regardless of species (e.g. Fe and Cu
    close enough to be alternate occupants of one metal slot still get
    grouped). At most one real atom can ever come from a group.

    radius_threshold is the tuning knob (1.0 = group when radii would
    literally overlap); species_radii overrides individual elements'
    tabulated radii. search_radius just bounds the pairwise-distance scan
    for speed -- must exceed the largest cutoff this call can produce.

    A species with an unusually large tabulated radius can chain far more
    sites together than physically intended (see _detect_site_groups_by_
    histogram for an alternative that doesn't depend on tabulated radii).

    Returns (groups, species_radii_used); the latter maps each dominant
    species among the disordered sites to the radius actually applied.
    """
    disordered = _disordered_site_indices(structure)
    dominant_species = {i: _dominant_species(structure[i]) for i in disordered}

    species_radii_used: Dict[str, float] = {}
    for sp in set(dominant_species.values()):
        species_radii_used[sp] = _species_radius(sp, species_radii)
    radii = {i: species_radii_used[sp] for i, sp in dominant_species.items()}

    parent = {i: i for i in disordered}
    touched: Set[int] = set()
    for d, i, j in _pairwise_distances(structure, disordered, search_radius):
        if d < radius_threshold * (radii[i] + radii[j]):
            touched.add(i)
            touched.add(j)
            _union(i, j, parent)

    groups: Dict[int, List[int]] = {}
    for i in touched:
        groups.setdefault(_find(i, parent), []).append(i)
    return {gid: sorted(members) for gid, members in groups.items()}, species_radii_used


def _auto_group_cutoff(
    structure: Structure, indices: Sequence[int],
    search_radius: float = _GROUP_SEARCH_RADIUS, jump_ratio: float = DEFAULT_JUMP_RATIO,
) -> Optional[float]:
    """Pick a group_cutoff (Å) from this structure's own geometry.

    Disorder-split spacings (alternate positions for one physical atom)
    cluster far tighter than any real bond. So: collect every pairwise
    distance among disordered sites under search_radius, sort the
    distinct values, and cut at the first jump whose ratio to the
    previous distance is at least jump_ratio -- the boundary right after
    the tight low-distance tier. Deliberately the FIRST such jump, not
    the largest gap anywhere in the range, since a second short-but-real
    tier (e.g. cross-bond contacts) could otherwise get merged in.

    jump_ratio=1.7 is a heuristic default, not a law of nature -- the
    printed cutoff exists so an unusual structure can be sanity-checked
    by eye and overridden with an explicit group_cutoff.

    Returns None only when no pair of disordered sites falls within
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
        gap = hi - lo
        if gap > best_gap:
            best_gap, best_cutoff = gap, (lo + hi) / 2
    return best_cutoff


def _detect_site_groups_by_histogram(
    structure: Structure,
    group_cutoff: Optional[float] = None,
    search_radius: float = _GROUP_SEARCH_RADIUS,
    jump_ratio: float = DEFAULT_JUMP_RATIO,
) -> Tuple[Dict[int, List[int]], float]:
    """Group disordered (occupancy < 1) sites: any pair closer than
    group_cutoff (Å) is unioned transitively into one group, regardless
    of species. At most one real atom can ever come from a group.

    group_cutoff=None (default) auto-derives it per structure from the
    pairwise-distance histogram (_auto_group_cutoff) rather than assuming
    a value tuned on a different structure still applies. Pass an
    explicit float only after checking it against this structure's own
    distance histogram (see the plot sat_solver writes).

    Returns (groups, cutoff_used).
    """
    disordered = _disordered_site_indices(structure)
    if group_cutoff is None:
        auto = _auto_group_cutoff(structure, disordered, search_radius, jump_ratio)
        group_cutoff = auto if auto is not None else 0.0

    parent = {i: i for i in disordered}
    touched: Set[int] = set()
    for d, i, j in _pairwise_distances(structure, disordered, group_cutoff):
        touched.add(i)
        touched.add(j)
        _union(i, j, parent)

    groups: Dict[int, List[int]] = {}
    for i in touched:
        groups.setdefault(_find(i, parent), []).append(i)
    return {gid: sorted(members) for gid, members in groups.items()}, group_cutoff


def _plot_group_cutoff_histogram(
    distances: Sequence[float], cutoff: Optional[float], cutoff_is_auto: bool, out_path: str
) -> None:
    """Histogram of the pairwise-distance population _auto_group_cutoff
    scans, with the resolved group_cutoff marked, so the disorder-split
    tier and the jump into real-bond distances are visible directly."""
    plt.figure(figsize=(8, 5))
    n_bins = min(60, max(10, len(set(round(d, 3) for d in distances))))
    plt.hist(distances, bins=n_bins, color="steelblue", edgecolor="black")
    if cutoff is not None:
        origin = "auto-derived" if cutoff_is_auto else "explicitly passed"
        plt.axvline(
            cutoff, color="red", linestyle="--", linewidth=2,
            label=f"group_cutoff = {cutoff:.4f} Å ({origin})",
        )
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
    """Real-bonding-shell neighbors for every site -- for checking rule/
    motif satisfaction, distinct from disorder-group detection: every
    disorder candidate is present at once in `structure`, so a naive
    neighbor search would be swamped by alternate positions of the same
    atom a fraction of an Angstrom apart.

    For each site i, runs CrystalNN(x_diff_weight=x_diff_weight) on a
    proxy structure excluding only i's own group-mates, so i sees an
    honest shell without being swamped by its own disorder-split
    alternates.

    x_diff_weight: how much CrystalNN trusts electronegativity to decide
    what's a bond. 0.0 (default) is pure geometry -- needed when the
    real distinction is same-element at two distances (e.g. ice's
    covalent vs. hydrogen-bonded O-H). Raise toward 1.0 (CrystalNN's own
    default) to favor a high-electronegativity-difference pair (e.g.
    Cu-S) over a same-element near-contact (e.g. Cu-Cu) that would
    otherwise be miscounted as a bond. No one value suits both cases.
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
        bonding_neighbor_cache[i] = [
            {"site_index": keep_indices[n["site_index"]]} for n in neighbors
        ]

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
    disorder groups (at most one real atom per group, or exactly one if
    the group is never vacant), independently occupancy-disordered sites,
    each site's real bonding-shell neighbors (_build_bonding_neighbor_
    cache), and the global stoichiometric quota each species must hit.

    method picks how disorder groups are detected:
    - "histogram" (default): _detect_site_groups_by_histogram, using
      group_cutoff (None auto-derives it) and jump_ratio.
    - "radii": _detect_site_groups_by_radii, using radius_threshold and
      species_radii.

    Group detection (whichever method) works directly on pairwise
    distances; ctx.neighbor_cache (for rule/motif satisfaction) is a
    separate CrystalNN pass over real bonding partners instead
    (_build_bonding_neighbor_cache, tuned via x_diff_weight). Conflating
    the two would make any rule needing a non-group-mate neighbor
    silently unsatisfiable.

    Pulled out of sat_solver() so cluster_infeasible_rules() reuses the
    same group definition sat_solver()'s own constraints are built from.
    """
    species_radii_used = None
    resolved_group_cutoff = None
    if method == "radii":
        groups, species_radii_used = _detect_site_groups_by_radii(structure, radius_threshold, species_radii)
    elif method == "histogram":
        groups, resolved_group_cutoff = _detect_site_groups_by_histogram(structure, group_cutoff, jump_ratio=jump_ratio)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'radii' or 'histogram'")
    group_of: Dict[int, int] = {i: gid for gid, members in groups.items() for i in members}

    # A group's members' occupancies, summed, say whether that slot is
    # ALWAYS occupied in the real crystal (total ~1) or only SOMETIMES
    # (total < 1, real vacancy disorder). Only the first gets a hard
    # "exactly one real atom" floor; the tolerance (not round-to-nearest)
    # avoids a total like 0.9 being wrongly treated as "always occupied".
    group_total_occ: Dict[int, float] = {
        gid: sum(occ for i in members for occ in structure[i].species.as_dict().values())
        for gid, members in groups.items()
    }
    group_mandatory: Set[int] = {
        gid for gid, total in group_total_occ.items() if total >= 1.0 - 1e-3
    }

    labels = [
        site.properties.get("_atom_site_label", f"{site.species_string}{i+1}")
        for i, site in enumerate(structure)
    ]

    # Majority-vote species per site, used only to give the bonding-
    # neighbor CrystalNN pass a single geometry to work with -- never used
    # to decide what's really there.
    geometry_species_of = [_dominant_species(site) for site in structure]

    # species_of: always-present, single-species, non-grouped sites -- no
    # CP-SAT variable needed. member_species: every togglable site's own
    # candidate species at its own coordinate (grouped or independently
    # disordered), tracked per site rather than per group so a later step
    # can tell which specific member of a group is real.
    species_of: Dict[int, str] = {}
    member_species: Dict[int, List[str]] = {}
    for i, site in enumerate(structure):
        occ_dict = site.species.as_dict()
        if i not in group_of and sum(occ_dict.values()) >= 1.0 - 1e-9:
            species_of[i] = _dominant_species(site)
        else:
            member_species[i] = sorted(occ_dict.keys())

    togglable: Set[int] = set(member_species)

    # Global stoichiometric quota per species: raw occupancy summed across
    # every togglable site that could realize it, spanning grouped and
    # non-grouped sites in one pass.
    species_raw_occ_total: Dict[str, float] = {}
    for i in togglable:
        for sp, occ in structure[i].species.as_dict().items():
            species_raw_occ_total[sp] = species_raw_occ_total.get(sp, 0.0) + occ
    stoichiometry = {sp: round(total) for sp, total in species_raw_occ_total.items()}

    # Every raw site index that could ever be species `sp`, precomputed
    # once since _sat_is_feasible calls _sat_build_model repeatedly.
    source_refs_by_species: Dict[str, List[int]] = {}
    for i, candidates in member_species.items():
        for sp in candidates:
            source_refs_by_species.setdefault(sp, []).append(i)
    for i, sp in species_of.items():
        source_refs_by_species.setdefault(sp, []).append(i)

    bonding_neighbor_cache = _build_bonding_neighbor_cache(
        structure, geometry_species_of, group_of, groups, x_diff_weight
    )

    return _SATContext(
        structure=structure,
        labels=labels,
        species_of=species_of,
        member_species=member_species,
        togglable=togglable,
        groups=groups,
        group_of=group_of,
        group_mandatory=group_mandatory,
        source_refs_by_species=source_refs_by_species,
        stoichiometry=stoichiometry,
        neighbor_cache=bonding_neighbor_cache,
        method=method,
        radius_threshold=radius_threshold if method == "radii" else None,
        species_radii_used=species_radii_used,
        group_cutoff=resolved_group_cutoff,
    )


def _neighbor_refs_by_species(src_lookup_i: int, ctx: _SATContext) -> Dict[str, List[int]]:
    """Groups the real neighbors cached at ctx.neighbor_cache[src_lookup_i]
    by ref (deduplicated, since a group can only ever supply one real
    atom) and then by every species that ref could turn out to be. Shared
    by _sat_build_model and cluster_infeasible_rules so both agree on
    what "a neighbor of species X" means."""
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
    contribute at most one real atom between them; a non-grouped site is
    independent. A fast necessary condition for "would this need 2+ real
    atoms from one group at once" -- not a full feasibility oracle (that's
    _sat_is_feasible/CP-SAT), and not joint across different target
    species in the same motif."""
    distinct_groups: Set[int] = set()
    independent = 0
    for ref in refs:
        gid = ctx.group_of.get(ref)
        if gid is not None:
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
    stoichiometry, no interaction between rules. Catches one specific
    unphysical case: a motif whose required neighbor count for some
    target species could only ever be met by getting 2+ real atoms out of
    one disorder group at once, which sat_solver()'s own <=1-per-group
    constraint can never allow.

    A rule is dropped only if this holds for every candidate source atom
    of its species -- one achievable atom is enough to keep it (sat_
    solver's own OR resolution handles a motif that only fires for some
    atoms of a species). `unmodeled` is separate from `dropped`: a motif
    can name a species that's not a candidate anywhere in this supercell,
    which has nothing to do with group geometry.

    method, radius_threshold, species_radii, group_cutoff, jump_ratio are
    passed through to _build_sat_context (see its docstring).

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

    for raw, (source_species, motif) in zip(rule_strings, parsed):
        if not motif:
            kept.append(raw)  # an "isolated" rule makes no neighbor-count claim, so nothing to check
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

        source_refs = ctx.source_refs_by_species.get(source_species, [])

        feasible_anywhere = False
        best_reason = None
        for src_ref in source_refs:
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
    counts, then uses SHAP to explain that model: each motif gets an
    importance (how much it matters) and a signed correlation (which
    direction it pushes the target). Writes plots and a CSV of the
    ranking to out_dir.

    Args:
        test_size: Fraction of structures held out for the test split.
        random_state: Seed for the split and the forest. None draws a
            fresh seed each call (printed either way, for reproducing a
            "random" run later).

    Returns the motif names sorted by that signed correlation.
    """
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
    print(importance.head(20))

    _plot_correlation_bar(importance, out_dir / "SHAP_Correlation.png")

    return importance.sort_values("Correlation")["Motif"].tolist()


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
               random_state: Optional[int] = None):
    """Resolve a disordered structure against a ranked list of candidate
    'Source-Motif' rules, then enumerate satisfying structures via CP-SAT.

    For each source species, candidate motifs are tried as a growing
    prefix (in list order) until the smallest jointly-satisfiable prefix
    is found; that prefix is granted (OR'd -- the atom must match at
    least one). Every motif is closed-world: a species not named in it is
    implicitly required to have zero neighbors, so "Fe-S4" means exactly
    4 S and nothing else, not merely "at least 4 S".

    method picks how disorder groups are detected (see _build_sat_context):
    "histogram" (default, group_cutoff/jump_ratio) or "radii"
    (radius_threshold/species_radii). With method="histogram" and
    plot_diagnostics on, also writes the distance histogram with the
    resolved cutoff marked.

    x_diff_weight (default 0.0): passed to CrystalNN when building the
    real bonding-shell neighbor cache that rules are checked against --
    see _build_bonding_neighbor_cache's docstring for what it controls
    and when to raise it.

    feasibility_time_limit: per-CP-SAT-call budget while resolving each
    species' motif prefix (not the final enumeration, which gets a fixed
    60s per solution). Lower trades certainty for speed -- flagged in the
    printed resolution line when a result is left unconfirmed.

    plot_diagnostics: writes one picture per distinct disorder-group
    signature under out_dir/_positional_clusters/. Solution CIFs go under
    out_dir/_sat_solutions/ regardless of this flag.

    randomize_solutions: plain solve-block-resolve enumeration tends to
    return each next solution as the nearest still-satisfying tweak of
    the last one. When True (default), each solve instead minimizes a
    fresh random linear objective over every toggle variable, so
    solutions are drawn from different parts of the feasible space.
    random_state seeds the draw (printed either way, so a spread can be
    reproduced); set randomize_solutions=False for plain enumeration.
    """
    structure = Structure.from_file(disordered_supercell_file)
    os.makedirs(out_dir, exist_ok=True)

    ctx = _build_sat_context(
        structure, method, radius_threshold, species_radii, group_cutoff, jump_ratio, x_diff_weight
    )

    if ctx.method == "radii":
        radii_str = ", ".join(f"{sp}={r:.3f} A" for sp, r in sorted(ctx.species_radii_used.items()))
        print(f"Site groups: radius_threshold={ctx.radius_threshold:g} x (r_i + r_j), radii {radii_str}")
    else:
        print(f"Site groups: histogram method, group_cutoff={ctx.group_cutoff:.4f} A")

    if not ctx.groups:
        print("  none")
    else:
        group_signatures = Counter()
        representative_gid = {}
        for gid, members in ctx.groups.items():
            species_tuple = tuple(sorted({sp for i in members for sp in ctx.member_species[i]}))
            signature = (len(members), species_tuple, gid in ctx.group_mandatory)
            group_signatures[signature] += 1
            representative_gid.setdefault(signature, gid)
        for (size, species_tuple, mandatory), n_groups in sorted(group_signatures.items()):
            desc = "/".join(species_tuple)
            kind = "==1" if mandatory else "<=1"
            print(f"  {n_groups}x size-{size} ({desc}, {kind})")

        if plot_diagnostics:
            group_plot_dir = os.path.join(out_dir, "_positional_clusters")
            # Emptied, not just created, so a stale picture from a
            # previous run never sits there looking current.
            if os.path.isdir(group_plot_dir):
                shutil.rmtree(group_plot_dir)
            os.makedirs(group_plot_dir, exist_ok=True)
            if ctx.method == "histogram":
                disordered = _disordered_site_indices(structure)
                dists = [d for d, _, _ in _pairwise_distances(structure, disordered, _GROUP_SEARCH_RADIUS)]
                _plot_group_cutoff_histogram(
                    dists, ctx.group_cutoff, group_cutoff is None,
                    os.path.join(group_plot_dir, "group_cutoff_histogram.png"),
                )
            for (size, species_tuple, mandatory), gid in representative_gid.items():
                members = ctx.groups[gid]
                desc = "-".join(species_tuple)
                kind = "mandatory" if mandatory else "optional"
                fname = f"group_size{size}_{desc}_{kind}.png"
                n_of_signature = group_signatures[(size, species_tuple, mandatory)]
                title = (
                    f"{n_of_signature}x size-{size} {'/'.join(species_tuple)} group ({kind}) "
                    f"[representative: {ctx.labels[gid]} et al.]"
                )
                _plot_disorder_group_pies(
                    structure, members, title,
                    os.path.join(group_plot_dir, fname),
                )
            print(f"  -> {len(representative_gid)} group image(s) written to {group_plot_dir}/")

    print("Site occupancy:")
    if not ctx.stoichiometry:
        print("  none")
    else:
        for sp, target in ctx.stoichiometry.items():
            n_candidates = len(ctx.source_refs_by_species.get(sp, []))
            print(f"  {sp}: {target} of {n_candidates} candidate sites")

    # Resolve one source species at a time, in the order it first
    # appears in `rules`.
    candidates = _parse_candidates(rules)
    species_order = []
    candidates_by_species = {}
    for source_species, motif in candidates:
        if source_species not in candidates_by_species:
            candidates_by_species[source_species] = []
            species_order.append(source_species)
        candidates_by_species[source_species].append(motif)

    print("\nResolving rules:")
    resolved = {}
    failed_species = []
    for source_species in species_order:
        motifs = candidates_by_species[source_species]
        n_motifs = len(motifs)

        def check(k, _resolved=resolved, _source_species=source_species, _motifs=motifs):
            trial = dict(_resolved)
            trial[_source_species] = _motifs[:k]
            return _sat_is_feasible(trial, ctx, time_limit=feasibility_time_limit)

        # Feasibility is monotonic in k (a longer prefix only adds OR
        # disjuncts), so binary search finds the smallest feasible k in
        # ~log2(n_motifs) solver calls instead of a linear scan. 'unknown'
        # (solver timeout) is treated conservatively as not-yet-feasible.
        uncertain = False
        status_n = check(n_motifs)
        if status_n == "unknown":
            uncertain = True
        if status_n != "feasible":
            accepted_k = None
        else:
            lo, hi = 1, n_motifs
            while lo < hi:
                mid = (lo + hi) // 2
                status_mid = check(mid)
                if status_mid == "unknown":
                    uncertain = True
                if status_mid == "feasible":
                    hi = mid
                else:
                    lo = mid + 1
            accepted_k = lo

        if accepted_k is None:
            failed_species.append(source_species)
            note = " (unconfirmed: some check timed out)" if uncertain else ""
            print(f"  {source_species}: unsatisfiable with any candidate motif -> unconstrained{note}")
        else:
            resolved[source_species] = motifs[:accepted_k]
            granted_str = " OR ".join(
                ("isolated" if not mo else ", ".join(f"{sp}{n}" for sp, n in mo.items()))
                for mo in motifs[:accepted_k]
            )
            note = " (upper bound: some check timed out unconfirmed)" if uncertain else ""
            print(f"  {source_species}: accepted [{granted_str}]{note}")

    # Final model: base constraints + all resolved species' requirements.
    # Enumerate solutions one at a time (solve -> block -> re-solve).
    model, pres, sp_choice = _sat_build_model(resolved, ctx)
    all_toggle_vars = list(pres.values()) + list(sp_choice.values())

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 8

    solutions_dir = os.path.join(out_dir, "_sat_solutions")
    if os.path.isdir(solutions_dir):
        shutil.rmtree(solutions_dir)
    os.makedirs(solutions_dir, exist_ok=True)

    rng = random.Random(_resolve_seed(random_state)) if randomize_solutions else None

    n_found = 0
    final_status = None
    while n_found < max_solutions:
        if rng is not None:
            # A fresh random objective each solve steers CP-SAT toward a
            # different vertex of the feasible region every time, instead
            # of the nearest one still allowed after blocking the last
            # solution. Blocking constraints (added below) still apply
            # regardless of the objective, so exhaustiveness detection
            # (final_status == INFEASIBLE) is unaffected.
            model.Minimize(sum(rng.randint(-1000, 1000) * v for v in all_toggle_vars))
        status = solver.Solve(model)
        final_status = status
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        n_found += 1

        # Every site keeps its own true coordinate -- fixed sites always,
        # togglable sites whichever of their candidate positions was
        # actually selected.
        species_out: List[str] = []
        coords_out = []
        for i in range(len(structure)):
            if i in ctx.species_of:
                species_out.append(ctx.species_of[i])
                coords_out.append(structure[i].frac_coords)
                continue
            if not solver.Value(pres[i]):
                continue
            candidates = ctx.member_species[i]
            sp = candidates[0] if len(candidates) == 1 else next(
                s for s in candidates if solver.Value(sp_choice[(i, s)])
            )
            species_out.append(sp)
            coords_out.append(structure[i].frac_coords)

        out_structure = Structure(structure.lattice, species_out, coords_out)
        out_structure.to(filename=os.path.join(solutions_dir, f"solution_{n_found}.cif"))

        diff_terms = [(1 - v) if solver.Value(v) else v for v in all_toggle_vars]
        model.Add(sum(diff_terms) >= 1)

    if n_found == 0:
        print(f"\nstatus: {solver.StatusName(final_status)}")
        print("No solution found - the resolved rules aren't jointly satisfiable.")
    elif final_status == cp_model.INFEASIBLE:
        print(f"\nstatus: exhaustive")
        print(f"Total solutions written: {n_found}")
    elif n_found >= max_solutions:
        print(f"\nstatus: stopped - max_solutions cap reached")
        print(f"Total solutions written: {n_found} (capped, may not be exhaustive)")
    else:
        print(f"\nstatus: stopped - a solve attempt timed out")
        print(f"Total solutions written: {n_found} (inconclusive)")