"""
SHAP analysis and SAT solver over motif-prevalence features.

Trains a RandomForestRegressor on motif prevalence vs. a target property
(e.g. MACE energy) and ranks motifs by their SHAP-feature correlation.

Motif and energy extraction now live in motifs.py — see
the example at the bottom of this file for how the two modules connect.
"""

# imports (shap analysis)
import secrets
from pathlib import Path
from typing import Optional, Sequence, Dict, List, Set
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# imports (sat solver)
import os
import re
from dataclasses import dataclass
from collections import Counter
from pymatgen.core import Structure
from pymatgen.core.local_env import MinimumDistanceNN
from ortools.sat.python import cp_model


# Figure export settings for publication-quality output
SAVE_KWARGS = dict(dpi=300, bbox_inches="tight")


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
    """First column = identifier (unused downstream), last = target, middle = features."""
    y = df.iloc[:, -1]
    X = df.iloc[:, 1:-1]

    X = X.loc[:, X.nunique() > 1]          # drop constant columns
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"Structures : {len(df)}")
    print(f"Features   : {X.shape[1]}")
    return X, y


def _resolve_seed(random_state: Optional[int]) -> int:
    """
    Resolve a seed for this run. Pass random_state=None for a truly random
    (OS-entropy-sourced) seed each call; an int pins a reproducible run.
    Either way the resolved seed is printed so a "random" run can be
    re-run exactly later by passing it back in as random_state.
    """
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
    """
    Pearson correlation between matching columns of two 2D arrays, vectorized
    across all columns at once (replaces a per-column np.corrcoef loop).
    """
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
    correlations = _pearson_corr_columns(X.to_numpy(dtype=float), shap_values)
    importance = pd.DataFrame(
        {
            "Motif": X.columns,
            "MeanAbsSHAP": np.abs(shap_values).mean(axis=0),
            "Correlation": correlations,
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

    plt.xlabel("Feature–SHAP Correlation", fontsize=14)
    plt.ylabel("Motif", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    _save_fig(out_path)


# Ancillary Functions (sat)
#------------------------------------------------------------------------------------------------------------

def _parse_candidates(rule_strings):
    """Parses plain 'Source-Motif' strings (no yes/no) into
    (source_species, motif_dict) tuples, preserving order."""
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


def _clean_species_and_occ(site):
    occ_dict = site.species.as_dict()
    sp = max(occ_dict, key=occ_dict.get)
    return sp, occ_dict[sp]

@dataclass
class _SATContext:
    """Bundles the read-only structural/occupancy data that the CP-SAT
    model builder needs. Built once in sat_solver() and threaded
    explicitly through every helper that needs it, instead of those
    helpers reaching for variables that happen to be sitting in an
    enclosing scope."""
    structure: Structure
    labels: List[str]
    species_of: List[str]
    togglable: Set[int]
    clusters: Dict[int, List[int]]
    occ_by_species: Dict[str, List[int]]
    stoichiometry: Dict[str, int]
    neighbor_cache: Dict[int, List[dict]]


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
    # ------------------------------------------------------------
    # Build a fresh CP-SAT model from scratch: the base hard structural/
    # stoichiometric constraints, plus an OR-favoured constraint for each
    # (species, motif-list) pair supplied in `resolved`.
    # ------------------------------------------------------------
    m = cp_model.CpModel()
    pres = {i: m.NewBoolVar(f"sel_{ctx.labels[i]}") for i in ctx.togglable}

    def sel(i):
        return pres[i] if i in pres else 1

    for members in ctx.clusters.values():
        m.Add(sum(pres[i] for i in members) <= 1)

    for sp, members in ctx.occ_by_species.items():
        m.Add(sum(pres[i] for i in members) == ctx.stoichiometry[sp])

    def build_eq(src, motif, neigh_by_species, tag):
        if not motif:
            all_neighbor_indices = [n["site_index"] for n in ctx.neighbor_cache[src]]
            count_expr = sum(sel(j) for j in all_neighbor_indices)
            eq = m.NewBoolVar(f"eq_{tag}_isolated")
            m.Add(count_expr == 0).OnlyEnforceIf(eq)
            m.Add(count_expr != 0).OnlyEnforceIf(eq.Not())
            return eq
        eq_parts = []
        for target_species, n in motif.items():
            candidates = neigh_by_species.get(target_species, [])
            count_expr = sum(sel(j) for j in candidates)
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
        source_indices = [i for i in range(len(ctx.structure)) if ctx.species_of[i] == source_species]
        for src in source_indices:
            src_present = pres.get(src, None)
            neigh_by_species = {}
            for n in ctx.neighbor_cache[src]:
                j = n["site_index"]
                neigh_by_species.setdefault(ctx.species_of[j], []).append(j)
            eq_favoured = [
                build_eq(src, mo, neigh_by_species, f"{ctx.labels[src]}_{k}")
                for k, mo in enumerate(motifs)
            ]
            at_least_one = eq_favoured[0] if len(eq_favoured) == 1 else None
            if at_least_one is None:
                at_least_one = m.NewBoolVar(f"anyfav_{ctx.labels[src]}")
                m.AddBoolOr(eq_favoured).OnlyEnforceIf(at_least_one)
                m.AddBoolAnd([e.Not() for e in eq_favoured]).OnlyEnforceIf(at_least_one.Not())
            if src_present is None:
                m.Add(at_least_one == 1)
            else:
                m.Add(at_least_one == 1).OnlyEnforceIf(src_present)
    return m, pres


def _sat_is_feasible(resolved, ctx: _SATContext):
    m, pres = _sat_build_model(resolved, ctx)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 8
    status = solver.Solve(m)
    return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)



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
    """
    Run a Shapley analysis over motif-prevalence features to rank preferred
    motifs by their SHAP correlation with formation energy.

    Args:
        test_size: Fraction of structures held out for the test split.
        random_state: Seed for the train/test split and the forest. Pass an
            int for a reproducible run, or None for a fresh, non-reproducible
            random seed each call (the resolved seed is printed either way).

    Returns the list of motif names sorted by SHAP-feature correlation.
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
               nn_tol: float = 0.3,
               max_solutions: int = 400,
               out_dir: str = "."):
    # Plain ordered list of candidate motifs - no yes/no needed.
    # For each source species, motifs are tried as a growing prefix
    # (in the order given) until the first prefix that is jointly
    # satisfiable (together with all structural/stoichiometric hard
    # constraints) is found. That prefix is granted "yes" (combined via
    # OR - the atom must match at least one of them); every motif not
    # granted is automatically excluded, since an atom whose actual
    # neighbor pattern matches none of the accepted motifs cannot satisfy
    # the "at least one" requirement.
    # initialisation
    structure = Structure.from_file(disordered_supercell_file)
    nn_finder = MinimumDistanceNN(tol=nn_tol)
    os.makedirs(out_dir, exist_ok=True)

    labels = [
        site.properties.get("_atom_site_label", f"{site.species_string}{i+1}")
        for i, site in enumerate(structure)
    ]

    species_of = []
    occupancy_of = []
    for site in structure:
        sp, occ = _clean_species_and_occ(site)
        species_of.append(sp)
        occupancy_of.append(occ)

    neighbor_cache = {i: nn_finder.get_nn_info(structure, i) for i in range(len(structure))}

    # ------------------------------------------------------------
    # Detect two kinds of disorder (unrelated to the rule-resolution
    # process below - these are always hard, always applied first):
    # (a) structural: same-species mutual nearest neighbors (split positions, e.g. ice H-H)
    # (b) occupancy: fractional occupancy with no structural partner (vacancy disorder, e.g. In2Te3)
    # ------------------------------------------------------------
    same_species_neighbors = {
        i: {n["site_index"] for n in neighbor_cache[i] if species_of[n["site_index"]] == species_of[i]}
        for i in range(len(structure))
    }
    edges = set()
    for i, neighs in same_species_neighbors.items():
        for j in neighs:
            if i in same_species_neighbors[j]:
                edges.add((min(i, j), max(i, j)))

    structural_togglable = set()
    for i, j in edges:
        structural_togglable.add(i)
        structural_togglable.add(j)

    occupancy_togglable = {
        i for i in range(len(structure))
        if occupancy_of[i] < 1.0 - 1e-9 and i not in structural_togglable
    }

    togglable = structural_togglable | occupancy_togglable
    togglable_list = sorted(togglable)

    parent = {i: i for i in structural_togglable}

    for i, j in edges:
        _union(i, j, parent)
    clusters = {}
    for i in structural_togglable:
        clusters.setdefault(_find(i, parent), []).append(i)

    occ_by_species = {}
    for i in occupancy_togglable:
        occ_by_species.setdefault(species_of[i], []).append(i)
    stoichiometry = {sp: round(sum(occupancy_of[i] for i in members)) for sp, members in occ_by_species.items()}

    print("Clusters (structural, hard <=1 constraint):")
    if not clusters:
        print("  none")
    else:
        cluster_signatures = Counter()
        for members in clusters.values():
            species_counts = Counter(species_of[i] for i in members)
            signature = (len(members), tuple(sorted(species_counts.items())))
            cluster_signatures[signature] += 1
        for (size, species_counts), n_clusters in sorted(cluster_signatures.items()):
            desc = ", ".join(f"{count}x {sp}" for sp, count in species_counts)
            print(f"  {n_clusters} cluster(s) of size {size}: {desc}")

    print("Occupancy-derived stoichiometry (hard, global count constraint):")
    if not stoichiometry:
        print("  none")
    else:
        for sp, target in stoichiometry.items():
            n_sites = len(occ_by_species[sp])
            print(f"  {sp}: {target} selected out of {n_sites} candidate sites")

    # ------------------------------------------------------------
    # Bundle everything the SAT-model builder needs into one context
    # object, and pass that explicitly to every helper that needs it
    # (rather than those helpers relying on it being present in an
    # enclosing scope).
    # ------------------------------------------------------------
    ctx = _SATContext(
        structure=structure,
        labels=labels,
        species_of=species_of,
        togglable=togglable,
        clusters=clusters,
        occ_by_species=occ_by_species,
        stoichiometry=stoichiometry,
        neighbor_cache=neighbor_cache,
    )

    # ------------------------------------------------------------
    # Incremental resolution, species by species, in order of first
    # appearance in RULES.
    # ------------------------------------------------------------
    candidates = _parse_candidates(rules)
    species_order = []
    candidates_by_species = {}
    for source_species, motif in candidates:
        if source_species not in candidates_by_species:
            candidates_by_species[source_species] = []
            species_order.append(source_species)
        candidates_by_species[source_species].append(motif)

    print("\nResolving rule sets (incremental satisfiability, in list order):")
    resolved = {}
    failed_species = []
    for source_species in species_order:
        motifs = candidates_by_species[source_species]
        accepted_k = None
        for k in range(1, len(motifs) + 1):
            trial = dict(resolved)
            trial[source_species] = motifs[:k]
            if _sat_is_feasible(trial, ctx):
                accepted_k = k
                break
        if accepted_k is None:
            failed_species.append(source_species)
            print(f"  {source_species}: NO feasible subset found, even using all {len(motifs)} "
                  f"candidate motifs (in combination with structural/stoichiometric constraints "
                  f"and any already-resolved species). This species is left UNCONSTRAINED by rules.")
        else:
            resolved[source_species] = motifs[:accepted_k]
            granted_str = " OR ".join(
                ("isolated (no neighbors)" if not mo else
                 ", ".join(f"{sp}{n}" for sp, n in mo.items()))
                for mo in motifs[:accepted_k]
            )
            print(f"  {source_species}: granted YES = [{granted_str}]  "
                  f"(motifs beyond index {accepted_k-1} in the list are excluded)")

    # ------------------------------------------------------------
    # Final model: base constraints + all resolved species' OR-favoured
    # requirements. Enumerate solutions one at a time (solve -> block ->
    # re-solve), up to max_solutions.
    # ------------------------------------------------------------
    model, presence = _sat_build_model(resolved, ctx)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 8

    n_found = 0
    final_status = None
    while n_found < max_solutions:
        status = solver.Solve(model)
        final_status = status
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        n_found += 1
        chosen = sorted(set(range(len(structure))) - togglable)
        selected_now = []
        for i in togglable_list:
            if solver.Value(presence[i]):
                chosen.append(i)
                selected_now.append(i)
        chosen.sort()
        species_out = [species_of[i] for i in chosen]
        coords_out = [structure[i].frac_coords for i in chosen]
        out_structure = Structure(structure.lattice, species_out, coords_out)
        out_structure.to(filename=os.path.join(out_dir, f"solution_{n_found}.cif"))

        selected_set = set(selected_now)
        diff_terms = [
            (1 - presence[i]) if i in selected_set else presence[i]
            for i in togglable_list
        ]
        model.Add(sum(diff_terms) >= 1)

    if n_found == 0:
        print(f"\nstatus: {solver.StatusName(final_status)}")
        print("No solution found - the resolved hard rules are not satisfiable together with "
              "the structural/stoichiometric constraints above.")
    elif final_status == cp_model.INFEASIBLE:
        print(f"\nstatus: exhaustive - no more solutions exist")
        print(f"Total solutions written: {n_found} (exhaustive)")
    elif n_found >= max_solutions:
        print(f"\nstatus: stopped - max_solutions cap reached")
        print(f"Total solutions written: {n_found} (capped - may not be exhaustive, "
              f"increase max_solutions to find more)")
    else:
        print(f"\nstatus: stopped - a solve attempt timed out (solver.max_time_in_seconds) "
              f"before confirming feasibility either way")
        print(f"Total solutions written: {n_found} (inconclusive - unclear if more solutions exist)")