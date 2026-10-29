"""Resource-funded hepatic preflight; no tumor growth, induction or deployment."""

import hashlib
import itertools
import json
import logging
from pathlib import Path
import warnings

import cobra
from cobra.flux_analysis import pfba
from cobra.util.array import create_stoichiometric_matrix
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAP = ROOT / "notebooks/serine_evidence_supply_map/config"
REGISTRY = ROOT / "notebooks/serine_spatial_bounded_transfer/field_registry.csv"
COUNTS = dict(basal=252, local=12)
MASS_G, FLUX_SCALE, VOLUME_NL = 1e-10, .1, 20.
FACTOR = MASS_G * FLUX_SCALE * 1e9  # pmol/(cell*h) per normalized flux
ATP, SER, WATER, OXYGEN = "CULTURE_ATPM", "EX_HC00068_s", "EX_HC00011_s", "EX_HC00017_s"
ARMS = {
    "export_closed": (0., 0., []),
    "inherited_ceiling": (.024, .2, []),
    "matched_unmodified_ceiling": (.056, .2, []),
    "matched_conditional_capacity": (.056, .056 / (12 * FACTOR), []),
    "conditional_PHGDH": (.056, .056 / (12 * FACTOR), ["r0338"]),
    "conditional_GLS": (.056, .056 / (12 * FACTOR), ["r0078"]),
    "conditional_PHGDH_SHMT": (.056, .056 / (12 * FACTOR), ["r0338", "r0230", "r0231"]),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields():
    f = pd.read_csv(REGISTRY)
    f = pd.concat([f, pd.DataFrame([dict(substrate="formate", cobra_exchange="",
        carbon_atoms=1, nitrogen_atoms=0, initial_uM=0.)])], ignore_index=True)
    f["initial_pmol"] = f.initial_uM * VOLUME_NL * .001
    assert len(f) == 21
    return f


def plan():
    rows = [dict(variant=v, policy=p, arm=a, horizon_h=h, atp_target=5., control="none", reverse=False)
            for v, p, a, h in itertools.product(["native_C1", "curated_transport"],
                ["sealed", "oxygen_accounted"], ARMS, [1., 12.])]
    for variant in ["native_C1", "curated_transport"]:
        for arm in ["export_closed", "matched_conditional_capacity"]:
            rows.append(dict(variant=variant, policy="oxygen_accounted", arm=arm, horizon_h=12.,
                atp_target=1., control="maintenance_sensitivity", reverse=False))
        for control, atp, arm in [("no_organic_carbon", 5., "export_closed"),
                                  ("no_resources_zero_demand", 0., "export_closed"),
                                  ("no_resources_positive_export", 0., "matched_conditional_capacity")]:
            rows.append(dict(variant=variant, policy="oxygen_accounted", arm=arm, horizon_h=12.,
                atp_target=atp, control=control, reverse=False))
    for arm in ["export_closed", "matched_conditional_capacity"]:
        rows.append(dict(variant="curated_transport", policy="oxygen_accounted", arm=arm,
            horizon_h=12., atp_target=5., control="merge_order", reverse=True))
    return [dict(case_id=i, **spec) for i, spec in enumerate(rows)]


def assemble(spec, f):
    joint = cobra.Model("hepatic_funded_source_budget")
    joint.solver = "glpk"
    joint.tolerance = 1e-9
    joint.solver.configuration.timeout = 60
    models, reactions = {}, {}
    tag = "map_02" if spec["variant"] == "native_C1" else "map_14"
    request, ceiling, knockouts = ARMS[spec["arm"]]
    order = ["local", "basal"] if spec["reverse"] else ["basal", "local"]
    for group in order:
        m = cobra.io.read_sbml_model(str(MAP / (tag + "_" + group + ".xml")))
        m.solver = "glpk"
        m.reactions.get_by_id(ATP).bounds = (spec["atp_target"], spec["atp_target"])
        m.reactions.get_by_id(SER).bounds = (0., ceiling if group == "local" else 0.)
        m.reactions.get_by_id(OXYGEN).upper_bound = 0.
        if group == "local":
            for rid in knockouts:
                m.reactions.get_by_id(rid).bounds = (0., 0.)
        allowed = set(f.cobra_exchange.dropna()) | {WATER}
        assert all(r.id in allowed or r.bounds == (0., 0.) for r in m.boundary)
        # Keep nominal transport ceilings; remove input for diagnostic controls.
        for _, row in f.iterrows():
            rid = row.cobra_exchange
            if not isinstance(rid, str) or rid not in m.reactions:
                continue
            organic = row.carbon_atoms > 0 and row.substrate not in ["bicarbonate", "CO2"]
            if spec["control"].startswith("no_resources") or (spec["control"] == "no_organic_carbon" and organic):
                m.reactions.get_by_id(rid).lower_bound = 0.
        models[group] = m.copy()
        for metabolite in list(m.metabolites):
            metabolite.id = group + "__" + metabolite.id
        for reaction in list(m.reactions):
            reaction.id = group + "__" + reaction.id
        m.repair()
        m.objective = m.problem.Objective(0, direction="min")
        joint.merge(m, inplace=True, objective="left")
        reactions[group] = {r.id: joint.reactions.get_by_id(group + "__" + r.id)
                            for r in models[group].reactions}
    gross, net = {}, {}
    for _, row in f.iterrows():
        name, rid = row.substrate, row.cobra_exchange
        gross[name], net[name] = 0., 0.
        for group in models:
            if rid in reactions[group]:
                r = reactions[group][rid]
                gross[name] += COUNTS[group] * FACTOR * r.reverse_variable
                net[name] += COUNTS[group] * FACTOR * r.flux_expression
        if name != "oxygen" or spec["policy"] == "sealed":
            joint.add_cons_vars(joint.problem.Constraint(gross[name], ub=float(row.initial_pmol / spec["horizon_h"]),
                                                         name="finite_stock_" + name))
    # A requirement constraint never overwrites the inherited ceiling.
    joint.add_cons_vars(joint.problem.Constraint(net["serine"], lb=request, ub=request, name="requested_serine"))
    joint.objective = joint.problem.Objective(gross["oxygen"], direction="min")
    return joint, models, reactions, gross, net


def solve(spec, f):
    joint, models, reactions, gross, net = assemble(spec, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        first = joint.optimize()
    row = spec | dict(status=first.status, requested_serine_pmol_h=ARMS[spec["arm"]][0],
        local_ceiling_normalized=ARMS[spec["arm"]][1])
    if first.status != "optimal":
        return row, [], [], [], []
    optimum = float(first.objective_value)
    sol = pfba(joint)
    vector_rows, bounds, qualities = [], [], []
    for group in sorted(models):
        m = models[group]
        vector = np.array([sol.fluxes[group + "__" + r.id] for r in m.reactions])
        sv = float(np.max(np.abs(create_stoichiometric_matrix(m) @ vector)))
        violation = max(max(r.lower_bound-v, v-r.upper_bound, 0.) for r, v in zip(m.reactions, vector))
        assert sv < 1e-7 and violation < 1e-7
        qualities.append(dict(case_id=spec["case_id"], group=group, max_Sv=sv, max_bound_violation=violation))
        for r, v in zip(m.reactions, vector):
            vector_rows.append(dict(case_id=spec["case_id"], group=group, reaction=r.id, flux_normalized=float(v)))
            bounds.append(dict(case_id=spec["case_id"], group=group, reaction=r.id, lb=r.lower_bound, ub=r.upper_bound))
    ledger = []
    for _, field in f.iterrows():
        name, rid = field.substrate, field.cobra_exchange
        positive, negative = 0., 0.
        for group in models:
            if rid in reactions[group]:
                v = float(sol.fluxes[group + "__" + rid]) * COUNTS[group] * FACTOR
                positive += max(0., v)
                negative += max(0., -v)
        change = (positive-negative)*spec["horizon_h"]
        external = max(0., negative*spec["horizon_h"]-field.initial_pmol) if name == "oxygen" else 0.
        remaining = field.initial_pmol+external+change
        assert remaining >= -1e-7
        ledger.append(dict(case_id=spec["case_id"], substrate=name, initial_pmol=field.initial_pmol,
            gross_uptake_pmol_h=negative, gross_release_pmol_h=positive, net_change_pmol=change,
            explicit_external_pmol=external, remaining_pmol=remaining,
            residue_pmol=remaining-field.initial_pmol-external-change))
    balance = pd.DataFrame(ledger).set_index("substrate")
    cn = (balance.net_change_pmol.to_numpy() @ f[["carbon_atoms", "nitrogen_atoms"]].to_numpy())
    assert np.max(np.abs(cn)) < 1e-7
    oxygen = float(balance.loc["oxygen", "gross_uptake_pmol_h"])
    assert abs(oxygen-optimum) < 1e-7
    row.update(min_oxygen_pmol_h=oxygen, external_oxygen_pmol=float(balance.loc["oxygen", "explicit_external_pmol"]),
        delivered_serine_pmol=float(balance.loc["serine", "net_change_pmol"]),
        total_ATPM_pmol_h=sum(COUNTS.values())*FACTOR*spec["atp_target"],
        carbon_residual_pmol=float(cn[0]), nitrogen_residual_pmol=float(cn[1]),
        objective_l1=float(sol.objective_value))
    return row, ledger, vector_rows, bounds, qualities


def main():
    HERE.mkdir(exist_ok=True)
    logging.getLogger("cobra").setLevel(logging.ERROR)
    cobra.Configuration().solver = "glpk"
    inputs = [REGISTRY, Path(__file__)] + [MAP / (tag + "_" + group + ".xml")
        for tag in ["map_02", "map_14"] for group in COUNTS]
    protected = [ROOT / p for p in ["config/PhysiCell_settings.xml", "config/cells.csv", "main.cpp",
        "custom_modules/custom.cpp", "Makefile", "addons/dFBA/src/dfba_Model.cpp", "addons/dFBA/src/dfba_intracellular.cpp",
        "config/hepatonet_pan_zonal_curated_fastcore_reduced.xml"]]
    previous = ROOT / "notebooks/serine_matched_delivery_2026_10_21/final_provenance.json"
    archived = json.loads(previous.read_text())
    protected_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    for p, h in archived["files_sha256"].items():
        assert sha(ROOT / p) == h, p
        protected_hashes[p] = h
    f, cases = fields(), plan()
    protocol = dict(scope="constant-flux resource certificate before spatial coupling; not a dynamic culture or tumor response",
        defined_before_results=True, cases=cases, counts=COUNTS, tumors_not_solved=7,
        volume_nL=VOLUME_NL, dry_mass_g_per_cell=MASS_G, flux_scale=FLUX_SCALE,
        physical_factor_pmol_cell_h=FACTOR, scale_calibrated=False,
        inherited_serine_ceiling=.2, inherited_supply_pmol_h=12*FACTOR*.2,
        prescribed_reference_pmol_h=.056, required_ceiling_for_reference=.056/(12*FACTOR),
        oxygen_policies="sealed: finite initial stock; oxygen_accounted: minimize uptake, debit explicit external oxygen beyond initial stock",
        finite_resource_rule="gross uptake * horizon <= initial stock for all non-oxygen fields; no reuse of newly secreted nutrients",
        objectives="fixed ATPM per hepatocyte and fixed requested export; minimum gross O2 then pFBA at same optimum",
        source_capacity_not_induced=True, no_growth=True, no_signalling=True, no_tumor_consumption=True,
        medium="same inherited diagnostic 21-field stocks; not a confirmed complete hepatocyte culture recipe",
        scope_CN="21 external fields; water freely exchanged and has no C/N; no ATP/NADPH bookkeeping sources introduced",
        limitations="constant vectors over interval, homogeneous groups, no intracellular storage, no transport kinetics or aqueous porosity; infeasible != cell death",
        tolerances=dict(Sv=1e-7, bounds=1e-7, stock_pmol=1e-7, carbon_nitrogen_pmol=1e-7),
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs}, protected_files=protected_hashes)
    (HERE / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    f.to_csv(HERE / "field_registry.csv", index=False)
    result, ledger, vectors, bounds, quality = [], [], [], [], []
    for spec in cases:
        row, l, v, b, q = solve(spec, f)
        result.append(row); ledger.extend(l); vectors.extend(v); bounds.extend(b); quality.extend(q)
        print(spec["case_id"], spec["variant"], spec["policy"], spec["arm"], spec["horizon_h"],
              row["status"], row.get("external_oxygen_pmol", ""), flush=True)
    for name, rows in [("results", result), ("inventory", ledger), ("flux_vectors", vectors),
                       ("scenario_bounds", bounds), ("solver_quality", quality)]:
        pd.DataFrame(rows).to_csv(HERE / (name+".csv"), index=False)
    assert all(sha(ROOT / p) == h for p, h in protected_hashes.items())
    outputs = [HERE / name for name in ["protocol.json", "field_registry.csv", "results.csv",
        "inventory.csv", "flux_vectors.csv", "scenario_bounds.csv", "solver_quality.csv"]]
    (HERE / "execution_manifest.json").write_text(json.dumps(dict(cases=len(cases),
        all_completed=True, protected_unchanged=True, solver="glpk", cobra_version=cobra.__version__,
        outputs_sha256={p.name:sha(p) for p in outputs}), indent=2)+"\n")


if __name__ == "__main__":
    main()
