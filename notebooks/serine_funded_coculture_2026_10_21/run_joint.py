"""Finite shared-pool hepatic production plus tumor tasks; diagnostic only."""

import argparse
import csv
import gzip
import json
import logging
from pathlib import Path
import sys
import time

import cobra
import numpy as np
import pandas as pd
from cobra.util.array import create_stoichiometric_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUDGET = ROOT / "notebooks/serine_funded_source_budget_2026_10_21"
sys.path.insert(0, str(BUDGET))
import run_budget as hepatic  # noqa: E402
sys.path.insert(0, str(ROOT / "notebooks/serine_tumor_onecarbon_capacity"))
from run_capacity import build, configure  # noqa: E402
sys.path.insert(0, str(ROOT / "notebooks/serine_tumor_finite_demand"))
from run_finite_demand import add_tasks  # noqa: E402

COUNTS = dict(basal=252, local=12, tumor=7)
FACTOR, RATE, HORIZON = hepatic.FACTOR, .024, 12.
PORTS = dict(serine="EX_SER_c", glycine="EX_GLY_c", formate="EX_FOR_c", phosphate="EX_PI_c")
TASKS = dict(serine="DM_SER_c", C1="DM_BOOKKEEP_C1", ATP="TASK_ATP")
REQUIREMENTS = np.array([.4, .2, .1])*7/50
NATIVE = ROOT / "config/hepatonet_original_13_08.xml"
PARKER = ROOT / "notebooks/serine_tumor_response_audit/uptake_evidence/patu8902_serine_net_flux.json"
KOS = dict(PHGDH=["r0338"], GLS=["r0078"], PHGDH_SHMT=["r0338", "r0230", "r0231"])


def plan(smoke=False):
    rows = []
    for variant in ["native_C1", "curated_transport"]:
        for oxygen in ["sealed", "feed3"]:
            prefix = f"{variant}_{oxygen}_"
            for arm in ["closed", "funded", "prescribed_nominal", "prescribed_replay"]:
                rows.append(dict(case=prefix+arm, variant=variant, oxygen=oxygen, arm=arm,
                    dt_h=.1, horizon_h=HORIZON, priority="source_first", knockout="none",
                    source_case=prefix+"funded" if arm == "prescribed_replay" else "",
                    control="none"))
    for variant in ["native_C1", "curated_transport"]:
        for knockout in KOS:
            rows.append(dict(case=f"{variant}_feed3_{knockout}", variant=variant, oxygen="feed3",
                arm="funded", dt_h=.1, horizon_h=HORIZON, priority="source_first",
                knockout=knockout, source_case="", control="enzyme"))
    for arm in ["closed", "funded", "prescribed_replay"]:
        rows.append(dict(case=f"curated_transport_feed3_{arm}_halfdt", variant="curated_transport",
            oxygen="feed3", arm=arm, dt_h=.05, horizon_h=HORIZON, priority="source_first", knockout="none",
            source_case="curated_transport_feed3_funded_halfdt" if arm == "prescribed_replay" else "",
            control="refinement"))
    rows.append(dict(case="curated_transport_feed3_task_first", variant="curated_transport",
        oxygen="feed3", arm="funded", dt_h=.1, horizon_h=HORIZON, priority="task_first",
        knockout="none", source_case="", control="priority"))
    for arm in ["closed", "funded"]:
        rows.append(dict(case="no_resources_"+arm, variant="curated_transport", oxygen="sealed", arm=arm,
            dt_h=.1, horizon_h=1., priority="source_first", knockout="none", source_case="", control="no_resources"))
    rows.append(dict(case="hepatocytes_only", variant="curated_transport", oxygen="feed3", arm="closed",
        dt_h=.1, horizon_h=HORIZON, priority="source_first", knockout="none", source_case="", control="hep_only"))
    if smoke:
        rows = [r | dict(case="smoke_"+r["arm"], horizon_h=.2,
            source_case="smoke_funded" if r["arm"] == "prescribed_replay" else "")
            for r in rows if r["variant"] == "curated_transport" and r["oxygen"] == "feed3"
            and r["control"] == "none"]
    return rows


def group_models(spec, template, q):
    result = {}
    tag = "map_02" if spec["variant"] == "native_C1" else "map_14"
    for group in ["basal", "local"]:
        m = cobra.io.read_sbml_model(str(hepatic.MAP / (tag+"_"+group+".xml")))
        m.solver = "glpk"
        m.reactions.get_by_id(hepatic.ATP).bounds = (0., 5.)
        m.reactions.get_by_id(hepatic.SER).bounds = (0., .2 if group == "local" and spec["arm"] == "funded" else 0.)
        if group == "local":
            for rid in KOS.get(spec["knockout"], []):
                m.reactions.get_by_id(rid).bounds = (0., 0.)
        if spec["control"] == "no_resources":
            for r in m.boundary:
                if r.id != hepatic.WATER:
                    r.lower_bound = max(0., r.lower_bound)
        result[group] = m
    m = configure(template, "cytosolic_C1", dict(serine_uM=100., glycine_uM=100., formate_uM=500.))
    scale = 7/50
    for r in m.reactions:
        r.bounds = tuple(v*scale for v in r.bounds)
    add_tasks(m, scale)
    for role, rid in PORTS.items():
        cap = q*7/1e6*1000 if role == "serine" else scale if role in ["glycine", "formate"] else 0.
        m.reactions.get_by_id(rid).bounds = (-cap, 1000*scale)
    for rid in ["EX_CO2_c", "EX_CO2_m", "EX_NH3_m", "EX_AKG_c", "EX_GLU_c", "EX_3PG_c"]:
        m.reactions.get_by_id(rid).bounds = (0., 0.)
    if spec["control"] == "hep_only":
        for r in m.reactions:
            r.bounds = (0., 0.)
    result["tumor"] = m
    return result


class JointSelector:
    def __init__(self, spec, fields, template, q):
        self.spec, self.fields = spec, fields
        self.groups = group_models(spec, template, q)
        model = cobra.Model("finite_shared_coculture")
        model.solver = "glpk"; model.tolerance = 1e-9
        model.solver.configuration.timeout = 60
        self.r, self.ids, self.S, self.factors = {}, {}, {}, {}
        for group, original in self.groups.items():
            copy = original.copy()
            self.ids[group] = original.reactions.list_attr("id")
            self.S[group] = create_stoichiometric_matrix(original)
            for m in list(copy.metabolites):
                m.id = group+"__"+m.id
            for r in list(copy.reactions):
                r.id = group+"__"+r.id
            copy.repair()
            # Copy reactions, not custom optlang constraints or objectives.
            model.add_reactions([r.copy() for r in copy.reactions])
            self.r[group] = {rid:model.reactions.get_by_id(group+"__"+rid) for rid in self.ids[group]}
            self.factors[group] = 1. if group == "tumor" else COUNTS[group]*FACTOR
        p = model.problem
        self.alpha, self.theta = p.Variable("HEP_SERVICE", lb=0., ub=1.), p.Variable("TUMOR_SERVICE", lb=0., ub=1.)
        model.add_cons_vars([self.alpha, self.theta])
        for group in ["basal", "local"]:
            model.add_cons_vars(p.Constraint(self.r[group][hepatic.ATP].flux_expression-5*self.alpha,
                                            lb=0., ub=0., name="maintenance_"+group))
        for (name, rid), required in zip(TASKS.items(), REQUIREMENTS):
            model.add_cons_vars(p.Constraint(self.r["tumor"][rid].flux_expression-required*self.theta,
                                            lb=0., name="tumor_"+name))
        self.export = self.factors["local"]*self.r["local"][hepatic.SER].flux_expression
        self.net, self.gross, self.budgets = {}, {}, {}
        for field in fields.itertuples():
            net, gross = 0., 0.
            for group in self.groups:
                rid = PORTS.get(field.substrate) if group == "tumor" else field.cobra_exchange
                if rid in self.r[group]:
                    r = self.r[group][rid]
                    net += self.factors[group]*r.flux_expression
                    gross += self.factors[group]*r.reverse_variable
            self.net[field.substrate], self.gross[field.substrate] = net, gross
            expression = gross-self.export if field.substrate == "serine" else gross
            constraint = p.Constraint(expression, ub=0., name="pool_"+field.substrate)
            model.add_cons_vars(constraint); self.budgets[field.substrate] = constraint
        self.faces = [p.Constraint(x, lb=0., name="preserve_"+name) for name, x in
                      [("hep", self.alpha), ("source", self.export), ("tumor", self.theta)]]
        self.oxygen_face = p.Constraint(self.gross["oxygen"], ub=None, name="preserve_O2")
        model.add_cons_vars(self.faces+[self.oxygen_face])
        self.L1 = sum(self.factors[g]*(r.forward_variable+r.reverse_variable)
                      for g in self.groups for r in self.r[g].values())
        self.model, self.calls = model, 0

    def solve(self, stock, dt, prescribed):
        for face in self.faces:
            face.lb = 0.
        self.oxygen_face.ub = None
        for field, amount in zip(self.fields.itertuples(), stock):
            self.budgets[field.substrate].ub = max(0., float(amount))/dt + (prescribed if field.substrate == "serine" else 0.)
        p, m = self.model.problem, self.model
        def optimize(expression, direction):
            m.objective = p.Objective(expression, direction=direction)
            value = m.slim_optimize(error_value=None)
            self.calls += 1
            if m.solver.status != "optimal":
                raise RuntimeError(m.solver.status)
            return float(value)
        alpha = min(1., max(0., optimize(self.alpha, "max")))
        self.faces[0].lb = alpha
        if self.spec["priority"] == "source_first":
            export = max(0., optimize(self.export, "max")); self.faces[1].lb = export
            theta = min(1., max(0., optimize(self.theta, "max"))); self.faces[2].lb = theta
        else:
            theta = min(1., max(0., optimize(self.theta, "max"))); self.faces[2].lb = theta
            export = max(0., optimize(self.export, "max")); self.faces[1].lb = export
        oxygen = max(0., optimize(self.gross["oxygen"], "min"))
        self.oxygen_face.ub = oxygen+1e-10
        optimize(self.L1, "min")
        vectors = {g:np.array([self.r[g][rid].flux for rid in self.ids[g]]) for g in self.groups}
        hnet, tnet, gross = [np.zeros(len(self.fields)) for _ in range(3)]
        sv, error = 0., 0.
        for group, vector in vectors.items():
            sv = max(sv, float(np.abs(self.S[group]@vector).max()))
            error = max(error, *(max(r.lower_bound-x, x-r.upper_bound, 0.)
                                 for r, x in zip(self.groups[group].reactions, vector)))
            for j, field in enumerate(self.fields.itertuples()):
                rid = PORTS.get(field.substrate) if group == "tumor" else field.cobra_exchange
                if rid in self.r[group]:
                    v = self.factors[group]*self.r[group][rid].flux
                    (tnet if group == "tumor" else hnet)[j] += v
                    gross[j] += max(0., -v)
        tasks = np.array([self.r["tumor"][rid].flux for rid in TASKS.values()])
        actual_alpha, actual_theta = float(self.alpha.primal), float(self.theta.primal)
        actual_export = self.factors["local"]*self.r["local"][hepatic.SER].flux
        assert max(sv, error, abs(actual_alpha-alpha), abs(actual_theta-theta), abs(actual_export-export)) < 1e-7
        assert np.max(REQUIREMENTS*actual_theta-tasks) < 1e-7
        stock_limit = np.array(stock)/dt
        stock_limit[self.fields.index[self.fields.substrate.eq("serine")][0]] += actual_export+prescribed
        assert np.max(gross-stock_limit) < 1e-7
        return dict(alpha=actual_alpha, theta=actual_theta, export=actual_export, tasks=tasks,
                    hnet=hnet, tnet=tnet, vectors=vectors, max_Sv=sv, bound_error=error,
                    oxygen=oxygen, BG_ATP=self.r["tumor"]["BG_ATP"].flux,
                    BG_NADPH=self.r["tumor"]["BG_NADPH"].flux)


def run_case(spec, fields, template, q):
    out = HERE / "outputs" / spec["case"]; out.mkdir(parents=True, exist_ok=True)
    selector = JointSelector(spec, fields, template, q)
    stock = fields.initial_pmol.to_numpy().copy()
    if spec["control"] == "no_resources":
        stock[:] = 0.
    initial = stock.copy(); hcum, tcum, ecum, roundoff = [np.zeros(len(fields)) for _ in range(4)]
    oidx = int(fields.index[fields.substrate.eq("oxygen")][0]); sidx = int(fields.index[fields.substrate.eq("serine")][0])
    oxygen_rate = 3. if spec["oxygen"] == "feed3" else 0.
    oxygen_donor = oxygen_rate*spec["horizon_h"]
    serine_donor = RATE*spec["horizon_h"] if spec["arm"].startswith("prescribed") else 0.
    replay = pd.read_csv(HERE / "outputs" / spec["source_case"] / "trajectory.csv") if spec["source_case"] else None
    hep_hours, tumor_hours, source_total, tasks_total, auc = 0., 0., 0., np.zeros(3), 0.
    records, inventories, times = [], [], []
    step_count = int(round(spec["horizon_h"]/spec["dt_h"]))
    assert abs(step_count*spec["dt_h"]-spec["horizon_h"]) < 1e-10
    if replay is not None:
        assert len(replay) == step_count
    with gzip.open(out / "flux_vectors.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        names = [g+"__"+rid for g in selector.groups for rid in selector.ids[g]]
        writer.writerow(["state_id", *names])
        start = time.perf_counter()
        for step in range(step_count):
            dt = spec["dt_h"]; before = stock.copy()
            first_feed = min(oxygen_donor, oxygen_rate*dt/2)
            oxygen_donor -= first_feed; stock[oidx] += first_feed; ecum[oidx] += first_feed
            prescribed = RATE if spec["arm"] == "prescribed_nominal" else float(replay.iloc[step].hep_export_pmol_h) if replay is not None else 0.
            prescribed = min(max(0., prescribed), serine_donor/dt)
            serine_donor -= prescribed*dt
            value = selector.solve(stock, dt, prescribed)
            writer.writerow([step, *[float(x) for g in selector.groups for x in value["vectors"][g]]])
            stock += dt*(value["hnet"]+value["tnet"])
            stock[sidx] += prescribed*dt; ecum[sidx] += prescribed*dt
            if stock.min() < -1e-7:
                raise RuntimeError("Negative finite pool")
            correction = np.maximum(0., stock)-stock; stock += correction; roundoff += correction
            last_feed = min(oxygen_donor, oxygen_rate*dt/2)
            oxygen_donor -= last_feed; stock[oidx] += last_feed; ecum[oidx] += last_feed
            hcum += value["hnet"]*dt; tcum += value["tnet"]*dt
            tasks_total += value["tasks"]*dt
            hep_hours += value["alpha"]*dt; tumor_hours += value["theta"]*dt
            source_total += value["export"]*dt
            auc += (before[sidx]+stock[sidx])/2/.02*dt
            residue = stock-initial-ecum-hcum-tcum-roundoff
            atoms = (stock-initial-ecum-roundoff) @ fields[["carbon_atoms", "nitrogen_atoms"]].to_numpy()
            atoms += np.array([3*tasks_total[0]+tasks_total[1], tasks_total[0]])
            assert max(np.abs(residue).max(), np.abs(atoms).max()) < 1e-7
            row = dict(state_id=step, time_h=(step+1)*dt, dt_h=dt, hep_service=value["alpha"],
                tumor_service=value["theta"], hep_service_h=hep_hours, tumor_service_h=tumor_hours,
                hep_export_pmol_h=value["export"], prescribed_pmol_h=prescribed,
                hep_export_total_pmol=source_total, prescribed_total_pmol=ecum[sidx],
                oxygen_external_pmol=ecum[oidx], oxygen_donor_remaining_pmol=oxygen_donor,
                serine_donor_remaining_pmol=serine_donor, serine_uM=stock[sidx]/.02,
                oxygen_uM=stock[oidx]/.02, serine_AUC_uM_h=auc,
                serine_task_pmol_h=value["tasks"][0], C1_task_pmol_h=value["tasks"][1], ATP_task_pmol_h=value["tasks"][2],
                tumor_BG_ATP_pmol_h=value["BG_ATP"], tumor_BG_NADPH_pmol_h=value["BG_NADPH"],
                max_Sv=value["max_Sv"], bound_error=value["bound_error"],
                max_pool_residue_pmol=float(np.abs(residue).max()), carbon_residue_pmol=atoms[0], nitrogen_residue_pmol=atoms[1])
            records.append(row)
            for j, field in enumerate(fields.itertuples()):
                inventories.append(dict(state_id=step, time_h=(step+1)*dt, substrate=field.substrate,
                    initial_pmol=initial[j], remaining_pmol=stock[j], external_pmol=ecum[j],
                    hepatic_net_pmol=hcum[j], tumor_net_pmol=tcum[j], roundoff_pmol=roundoff[j], residue_pmol=residue[j]))
        seconds = time.perf_counter()-start
    pd.DataFrame(records).to_csv(out / "trajectory.csv", index=False)
    pd.DataFrame(inventories).to_csv(out / "inventory.csv", index=False)
    summary = spec | dict(hep_service_h=hep_hours, tumor_service_h=tumor_hours,
        hep_export_pmol=source_total, prescribed_serine_pmol=float(ecum[sidx]), external_oxygen_pmol=float(ecum[oidx]),
        final_serine_uM=stock[sidx]/.02, serine_AUC_uM_h=auc, minimum_hep_service=min(r["hep_service"] for r in records),
        minimum_tumor_service=min(r["tumor_service"] for r in records), lp_calls=selector.calls, seconds=seconds)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(spec["case"], "hep-h", round(hep_hours, 5), "tumor-h", round(tumor_hours, 5),
          "hep-Ser", round(source_total, 6), "external-Ser", round(ecum[sidx], 6), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    (HERE / "outputs").mkdir(parents=True, exist_ok=True)
    logging.getLogger("cobra").setLevel(logging.ERROR); cobra.Configuration().solver = "glpk"
    prior = json.loads((BUDGET / "protocol.json").read_text())
    protected = dict(prior["protected_files"])
    for p in BUDGET.iterdir():
        if p.is_file(): protected[str(p.relative_to(ROOT))] = hepatic.sha(p)
    assert all(hepatic.sha(ROOT / p) == h for p, h in protected.items())
    fields = hepatic.fields(); specs = plan(args.smoke)
    inputs = [Path(__file__), BUDGET / "run_budget.py", hepatic.REGISTRY, NATIVE, PARKER,
        ROOT / "notebooks/serine_tumor_onecarbon_capacity/run_capacity.py",
        ROOT / "notebooks/serine_tumor_finite_demand/run_finite_demand.py"]
    inputs += [hepatic.MAP / (tag+"_"+g+".xml") for tag in ["map_02", "map_14"] for g in ["basal", "local"]]
    protocol = dict(specs=specs, defined_before_results=True, scope="well-mixed diagnostic coculture; not growth or induction",
        counts=COUNTS, volume_nL=20., hepatic_factor=FACTOR, hepatic_export_ceiling=.2, max_source_pmol_h=RATE,
        feed3="finite O2 reservoir at 3 pmol/h, 36 pmol over 12h; uncalibrated scenario above previous minimum budget",
        selection="max common hepatic ATPM fraction; max source or tumor tasks by priority; remaining criterion; min O2; physical-flux-weighted L1",
        resource_rule="single shared gross budget per substrate; only hepatic/prescribed Ser can be used contemporaneously; other releases become available next step",
        prescribed_matching="replay actual funded hepatic Ser rate at each corresponding step; source is an explicitly added Ser reservoir and bypasses hepatic production cost",
        tumor_energy="formal ATP/NADPH/NADH budgets retained, NOT funded by glucose/O2; no full tumor metabolic reconstruction",
        no_hepatic_serine_uptake=True, no_growth=True, no_signalling=True, no_spatial_transport=True,
        fractions_not_viability=True, presumed_producers_not_measured_induction=True,
        predeclared_tolerances=dict(pool_pmol=1e-7, Sv=1e-7, refinement_contrast_h=.05, replay_dose_pmol=1e-9),
        inputs_sha256={str(p.relative_to(ROOT)):hepatic.sha(p) for p in inputs}, protected_files=protected)
    suffix = "_smoke" if args.smoke else ""
    (HERE / ("protocol"+suffix+".json")).write_text(json.dumps(protocol, indent=2)+"\n")
    fields.to_csv(HERE / "field_registry.csv", index=False)
    q = json.loads(PARKER.read_text())["positive_net_consumption_nmol_per_million_cells_h"]
    template = build(cobra.io.read_sbml_model(str(NATIVE)))
    summaries = [run_case(spec, fields, template, q) for spec in specs]
    pd.DataFrame(summaries).to_csv(HERE / ("summary"+suffix+".csv"), index=False)
    assert all(hepatic.sha(ROOT / p) == h for p, h in protected.items())
    outputs = {str(p.relative_to(HERE)):hepatic.sha(p) for s in specs
               for p in (HERE / "outputs" / s["case"]).iterdir() if p.is_file()}
    (HERE / ("execution_manifest"+suffix+".json")).write_text(json.dumps(dict(
        cases=len(specs), protected_unchanged=True, outputs_sha256=outputs), indent=2)+"\n")


if __name__ == "__main__":
    main()
