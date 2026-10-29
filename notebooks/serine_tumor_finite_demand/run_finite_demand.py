"""Isolated finite-stock Ser/Gly/formate task test; not a tumor growth model."""

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import cobra
import numpy as np
from cobra.flux_analysis import pfba
from cobra.util.array import create_stoichiometric_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "notebooks/serine_tumor_onecarbon_capacity"))
from run_capacity import build, configure  # noqa: E402

OUT = HERE / "outputs"
PROTOCOL = ROOT / "notebooks/serine_shin2026_response_audit/external_challenge_protocol.json"
NATIVE = ROOT / "config/hepatonet_original_13_08.xml"
VARIANTS = ("uptake_only", "cytosolic_C1", "plus_mito_GCS", "plus_formate_oxidation")
VOLUMES_NL = (100.0, 1000.0)
DT_H = 2.0
END_H = 88.0
UPTAKE_CEILING_PMOL_H = 1.0
TASK = {"serine": 0.4, "one_carbon": 0.2, "atp_maintenance": 0.1}
STOCK_TO_PMOL = 0.001  # 1 uM * 1 nL = 0.001 pmol
ROLES = {"serine": "EX_SER_c", "glycine": "EX_GLY_c", "formate": "EX_FOR_c"}
TOL = 1e-7


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_tasks(model, capacity_scale=1.0):
    # C1 is a formal bookkeeping product; this is not a nucleotide reaction.
    c1 = cobra.Metabolite("BOOKKEEP_C1_c", name="bookkeeping one-carbon unit", compartment="c")
    model.add_metabolites([c1])
    c1_service = cobra.Reaction("TASK_C1")
    c1_service.add_metabolites({
        model.metabolites.get_by_id("HC00140_c"): -1,
        model.metabolites.get_by_id("HC00102_c"): 1,
        c1: 1,
    })
    c1_service.bounds = (0, 1000 * capacity_scale)
    model.add_reactions([c1_service])
    c1_sink = cobra.Reaction("DM_BOOKKEEP_C1")
    c1_sink.add_metabolites({c1: -1})
    c1_sink.bounds = (0, 1000 * capacity_scale)
    model.add_reactions([c1_sink])
    atp = cobra.Reaction("TASK_ATP")
    atp.add_metabolites({
        model.metabolites.get_by_id("HC00012_c"): -1,
        model.metabolites.get_by_id("HC00011_c"): -1,
        model.metabolites.get_by_id("HC00018_c"): 1,
        model.metabolites.get_by_id("HC00019_c"): 1,
    })
    atp.bounds = (0, 1000 * capacity_scale)
    model.add_reactions([atp])
    theta = model.problem.Variable("TASK_SERVICE", lb=0, ub=1)
    model.add_cons_vars([theta])
    for rid, required in (("DM_SER_c", TASK["serine"]),
                          ("DM_BOOKKEEP_C1", TASK["one_carbon"]),
                          ("TASK_ATP", TASK["atp_maintenance"])):
        model.add_cons_vars([model.problem.Constraint(
            model.reactions.get_by_id(rid).flux_expression - required * capacity_scale * theta,
            lb=0, name="minimum_" + rid,
        )])
    model.objective = model.problem.Objective(theta, direction="max")
    return model


def solve_step(template, variant, condition, stock, dt_h,
               uptake_ceiling_pmol_h=UPTAKE_CEILING_PMOL_H, capacity_scale=1.0):
    """Scale extensive flux bounds, tasks and supplied reference ceilings together."""
    if not np.isfinite(capacity_scale) or capacity_scale <= 0 or dt_h <= 0:
        raise ValueError("Positive capacity scale and time step required")
    model = configure(template, variant, condition)
    if capacity_scale != 1.0:
        for reaction in model.reactions:
            reaction.bounds = tuple(value * capacity_scale for value in reaction.bounds)
    if variant == "plus_formate_oxidation":
        # This arm must obtain NADPH from formate oxidation, not the background.
        model.reactions.BG_NADPH.bounds = (0, 0)
    for role, rid in ROLES.items():
        ceiling = (uptake_ceiling_pmol_h[role] if isinstance(uptake_ceiling_pmol_h, dict)
                   else uptake_ceiling_pmol_h)
        if not np.isfinite(ceiling) or ceiling < 0:
            raise ValueError("Nonnegative finite uptake ceiling required")
        cap = min(ceiling * capacity_scale, max(0.0, stock[role]) / dt_h)
        # An absent supplement closes uptake initially, but intracellular
        # production may still be released and enter the next interval's stock.
        model.reactions.get_by_id(rid).bounds = (-cap, 1000 * capacity_scale)
    add_tasks(model, capacity_scale)
    best = model.slim_optimize(error_value=None)
    assert model.solver.status == "optimal" and -TOL <= best <= 1 + TOL
    solution = pfba(model)
    assert solution.status == "optimal"
    flux = solution.fluxes
    theta = float(model.solver.variables["TASK_SERVICE"].primal)
    assert abs(theta - best) < 1e-6, (theta, best)
    assert np.max(np.abs(create_stoichiometric_matrix(model) @ flux.to_numpy())) < TOL
    for rid, requirement in (("DM_SER_c", TASK["serine"]),
                             ("DM_BOOKKEEP_C1", TASK["one_carbon"]),
                             ("TASK_ATP", TASK["atp_maintenance"])):
        assert float(flux[rid]) + TOL >= requirement * capacity_scale * theta, (rid, theta, flux[rid])
    net = {role: float(flux[rid]) * dt_h for role, rid in ROLES.items()}
    for role in ROLES:
        assert -net[role] <= stock[role] + TOL, (role, net[role], stock[role])
    return theta, flux, net


def simulate(template, condition, variant, volume_nl):
    initial = {role: condition[role + "_uM"] * volume_nl * STOCK_TO_PMOL for role in ROLES}
    stock = initial.copy()
    uptake = {role: 0.0 for role in ROLES}
    release = {role: 0.0 for role in ROLES}
    trajectory = []
    for step in range(int(round(END_H / DT_H))):
        theta, flux, net = solve_step(template, variant, condition, stock, DT_H)
        for role in ROLES:
            stock[role] += net[role]
            uptake[role] += max(0.0, -net[role])
            release[role] += max(0.0, net[role])
            if abs(stock[role]) < TOL:
                stock[role] = 0.0
        trajectory.append(dict(
            condition=condition["id"], variant=variant, volume_nL=volume_nl,
            time_h=(step + 1) * DT_H, service_fraction=theta,
            serine_service_pmol_h=float(flux["DM_SER_c"]),
            C1_service_pmol_h=float(flux["DM_BOOKKEEP_C1"]),
            ATP_task_pmol_h=float(flux["TASK_ATP"]),
            ATP_background_pmol_h=float(flux["BG_ATP"]),
            NADPH_background_pmol_h=float(flux["BG_NADPH"]),
            **{"uptake_" + role + "_pmol": max(0.0, -net[role]) for role in ROLES},
            **{"release_" + role + "_pmol": max(0.0, net[role]) for role in ROLES},
            **{"remaining_" + role + "_pmol": stock[role] for role in ROLES},
        ))
    balance = {role: initial[role] + release[role] - uptake[role] - stock[role] for role in ROLES}
    assert all(abs(x) < TOL for x in balance.values()), balance
    assert all(x >= -TOL for x in stock.values())
    summary = dict(
        condition=condition["id"], variant=variant, volume_nL=volume_nl,
        service_hours_equivalent=sum(row["service_fraction"] * DT_H for row in trajectory),
        mean_service_fraction=sum(row["service_fraction"] for row in trajectory) / len(trajectory),
        final_service_fraction=trajectory[-1]["service_fraction"],
        first_time_below_full_h=next((row["time_h"] for row in trajectory
                                      if row["service_fraction"] < 1 - 1e-6), None),
        **{"initial_" + role + "_pmol": initial[role] for role in ROLES},
        **{"total_uptake_" + role + "_pmol": uptake[role] for role in ROLES},
        **{"total_release_" + role + "_pmol": release[role] for role in ROLES},
        **{"final_" + role + "_pmol": stock[role] for role in ROLES},
        max_abs_stock_residual_pmol=max(abs(x) for x in balance.values()),
    )
    return summary, trajectory


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    logging.getLogger("cobra").setLevel(logging.ERROR)
    cobra.Configuration().solver = "glpk"
    OUT.mkdir(exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    native = cobra.io.read_sbml_model(str(NATIVE))
    template = build(native)
    results, courses = [], []
    for volume in VOLUMES_NL:
        for variant in VARIANTS:
            for condition in protocol["conditions"]:
                result, rows = simulate(template, condition, variant, volume)
                results.append(result)
                courses.extend(rows)
    write_csv(OUT / "finite_task_summary.csv", results)
    write_csv(OUT / "finite_task_timecourses.csv", courses)
    # Chemical controls are recorded as qualitative checks, without fitting rates to them.
    controls = []
    for volume in VOLUMES_NL:
        for variant in VARIANTS:
            part = {r["condition"]: r for r in results
                    if r["volume_nL"] == volume and r["variant"] == variant}
            controls.append(dict(volume_nL=volume, variant=variant,
                serine_above_EAA=part["EAA_SER"]["service_hours_equivalent"] > part["EAA"]["service_hours_equivalent"] + TOL,
                gly_formate_above_EAA=part["EAA_GLY_FOR"]["service_hours_equivalent"] > part["EAA"]["service_hours_equivalent"] + TOL,
                gly_alone_above_EAA=part["EAA_GLY"]["service_hours_equivalent"] > part["EAA"]["service_hours_equivalent"] + TOL,
                formate_alone_above_EAA=part["EAA_FOR"]["service_hours_equivalent"] > part["EAA"]["service_hours_equivalent"] + TOL))
    write_csv(OUT / "chemical_control_comparison.csv", controls)
    assert len(results) == len(VOLUMES_NL) * len(VARIANTS) * len(protocol["conditions"])
    assert all(r["max_abs_stock_residual_pmol"] < TOL for r in results)
    manifest = dict(
        scope="Isolated finite-stock diagnostic task, no biomass or confluence prediction",
        source_protocol="Shin 2026 S5E nominal additions, not measured time courses of media",
        total_cases=len(results), time_steps_per_case=int(END_H / DT_H),
        assumed_volumes_nL=list(VOLUMES_NL), dt_h=DT_H,
        normalized_intrinsic_uptake_ceiling_pmol_h=UPTAKE_CEILING_PMOL_H,
        task_pmol_h=TASK,
        medium_replenishment="zero in this diagnostic; experimental exchanges not characterized",
        concentration_to_stock="uM * nL * 0.001 = pmol",
        background_ATP_NADPH_NADH_budgets="ATP and NADH each at most 1 normalized unit/h; NADPH at most 1 except formate-oxidation arm, where background NADPH is closed",
        formate_oxidation_arm="background NADPH budget closed; r0227 supplies it from formate",
        finite_media_stocks="Ser/Gly/formate only; CO2/NH3 disposal and non-organic pools remain open",
        C1_task="formal one-carbon withdrawal; not nucleotide synthesis",
        response_interpretation="task service fraction, not division, survival, viable counts or confluence",
        no_growth_calibration=True,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       [PROTOCOL, NATIVE, ROOT / "notebooks/serine_tumor_onecarbon_capacity/run_capacity.py"]},
        output_sha256={p.name: sha(p) for p in OUT.glob("*.csv")},
    )
    (HERE / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("cases", len(results), "time rows", len(courses))
    for row in results:
        if row["volume_nL"] == VOLUMES_NL[0] and row["variant"] == "cytosolic_C1":
            print(row["condition"], "service-h", round(row["service_hours_equivalent"], 3),
                  "final service", round(row["final_service_fraction"], 3))


if __name__ == "__main__":
    main()
