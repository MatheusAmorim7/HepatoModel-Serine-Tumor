"""Resource-paid energy prototype. Lumped C/N chemistry, NOT PATU calibration."""

from pathlib import Path
import sys

import cobra
import numpy as np
from cobra.util.array import create_stoichiometric_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "notebooks/serine_tumor_onecarbon_capacity"))
from run_capacity import build, configure, cn
sys.path.insert(0, str(ROOT / "notebooks/serine_tumor_finite_demand"))
from run_finite_demand import add_tasks

NATIVE = ROOT / "config/hepatonet_original_13_08.xml"
PARKER = ROOT / "notebooks/serine_tumor_response_audit/uptake_evidence/patu8902_serine_net_flux.json"
PREVIOUS = ROOT / "notebooks/serine_banh_respiration_audit_2026_10_21"
ROLES = dict(serine="EX_SER_c", glycine="EX_GLY_c", formate="EX_FOR_c",
             glucose="EX_E_GLC", oxygen="EX_E_O2", lactate="EX_E_LAC",
             CO2="EX_CO2_c", phosphate="EX_PI_c")
ATOMS = dict(serine=(3, 1), glycine=(2, 1), formate=(1, 0), glucose=(6, 0),
             oxygen=(0, 0), lactate=(3, 0), CO2=(1, 0), phosphate=(0, 0))
MAINTENANCE, WORK_ATP = .0035, .0105
SER_TASK, C1_TASK = .056, .028
BACKGROUND = ["BG_ATP", "BG_NADPH", "BG_NADH_c", "BG_NADH_m"]
NEW_CN = dict(E_GLC_c=(6, 0), E_O2_c=(0, 0), E_LAC_c=(3, 0), BOOKKEEP_C1_c=(1, 0))


def atoms(metabolite):
    return np.array(NEW_CN[metabolite.id] if metabolite.id in NEW_CN else cn(metabolite))


def add(model, rid, stoich):
    reaction = cobra.Reaction(rid, lower_bound=0., upper_bound=140.)
    reaction.add_metabolites({model.metabolites.get_by_id(k):v for k, v in stoich.items()})
    reaction.notes["scope"] = "Effective energy bookkeeping; ATP yield hypothetical; no complete proton/charge or thermodynamic validation"
    model.add_reactions([reaction])
    return reaction


class PaidSelector:
    def __init__(self, template, spec, q_serine):
        m = configure(template, "cytosolic_C1", dict(serine_uM=100., glycine_uM=100., formate_uM=500.))
        for reaction in m.reactions:
            reaction.bounds = tuple(v*7/50 for v in reaction.bounds)
        add_tasks(m, 7/50)
        m.id = "tumor_tasks_RESOURCE_PAID_DIAGNOSTIC_ONLY"
        m.notes["background"] = "All four formal ATP/redox ports closed. Glucose pays energy through two lumped routes. Redox reconstruction pending."
        m.notes["scope"] = "Seven equivalent tumor cells, fixed population; not a calibrated PATU-8902 reconstruction, growth or viability"
        for rid in BACKGROUND: m.reactions.get_by_id(rid).bounds = (0., 0.)
        for rid in ["EX_CO2_m", "EX_NH3_m", "EX_3PG_c", "EX_GLU_c", "EX_AKG_c"]:
            m.reactions.get_by_id(rid).bounds = (0., 0.)
        for rid, label in [("E_GLC_c", "Glucose"), ("E_O2_c", "Oxygen"), ("E_LAC_c", "Lactate")]:
            m.add_metabolites([cobra.Metabolite(rid, name=label+" lumped energy equivalent", compartment="c")])
        for rid, mid in [("EX_E_GLC", "E_GLC_c"), ("EX_E_O2", "E_O2_c"), ("EX_E_LAC", "E_LAC_c")]:
            add(m, rid, {mid:-1})
        # Neutral-equivalent lumping: glycolysis + LDH, or full oxidation.
        add(m, "PAID_GLYCOLYSIS", dict(E_GLC_c=-1, HC00018_c=-2, HC00019_c=-2,
            E_LAC_c=2, HC00012_c=2, HC00011_c=2))
        y = float(spec["ATP_per_glucose_oxidative"])
        if not np.isfinite(y) or y <= 0: raise ValueError("Positive oxidative yield required")
        add(m, "PAID_OXIDATION", dict(E_GLC_c=-1, E_O2_c=-6, HC00018_c=-y,
            HC00019_c=-y, HC00021_c=6, HC00012_c=y, HC00011_c=6+y))
        if spec.get("energy_routes_closed"):
            for rid in ["PAID_GLYCOLYSIS", "PAID_OXIDATION"]: m.reactions.get_by_id(rid).bounds = (0, 0)
        m.solver = "glpk"
        m.tolerance = 1e-9
        m.solver.configuration.timeout = 30
        p = m.problem
        self.theta = m.solver.variables["TASK_SERVICE"]
        if spec.get("work_disabled"): self.theta.ub = 0.
        self.alpha = p.Variable("MAINTENANCE_FRACTION", lb=0, ub=1)
        m.add_cons_vars(self.alpha)
        m.remove_cons_vars(m.solver.constraints["minimum_TASK_ATP"])
        m.add_cons_vars(p.Constraint(m.reactions.TASK_ATP.flux_expression-
            MAINTENANCE*self.alpha-WORK_ATP*self.theta, lb=0., ub=0., name="paid_energy_requirement"))
        m.add_cons_vars(p.Constraint(self.theta-self.alpha, ub=0., name="work_bounded_by_maintenance"))
        for rid in ["DM_SER_c", "DM_BOOKKEEP_C1"]:
            m.solver.constraints["minimum_"+rid].ub = 0.
        self.ex = {role:m.reactions.get_by_id(rid) for role, rid in ROLES.items()}
        self.caps = dict.fromkeys(ROLES, 0.)
        self.caps.update(serine=q_serine*7/1e6*1000, glycine=.14, formate=.14, glucose=.14, oxygen=.14)
        self.faces = {}
        for name, expression in dict(maintenance=self.alpha, work=self.theta,
            glucose=-self.ex["glucose"].flux_expression, oxygen=-self.ex["oxygen"].flux_expression).items():
            face = p.Constraint(expression, name="preserve_"+name)
            m.add_cons_vars(face)
            self.faces[name] = face
        self.model, self.spec = m, spec
        self.ids = m.reactions.list_attr("id")
        self.S = create_stoichiometric_matrix(m)
        self.L1 = sum(r.forward_variable+r.reverse_variable for r in m.reactions)
        self.calls = 0
        for reaction in m.reactions:
            if reaction.boundary: continue
            balance = sum((atoms(met)*value for met, value in reaction.metabolites.items()), start=np.zeros(2))
            assert np.max(np.abs(balance)) < 1e-10, (reaction.id, balance)

    def solve(self, stock, dt, interval=False):
        if dt <= 0 or any(not np.isfinite(v) or v < -1e-9 for v in stock.values()):
            raise ValueError("Finite nonnegative stocks and positive dt required")
        for face in self.faces.values():
            face.lb = None
            face.ub = None
        for role, reaction in self.ex.items():
            reaction.bounds = (-min(self.caps[role], max(0., stock[role])/dt), 140.)
            if role in ["glucose", "oxygen"]: reaction.upper_bound = 0.
        m, p = self.model, self.model.problem
        def optimize(expression, direction):
            m.objective = p.Objective(expression, direction=direction)
            value = float(m.slim_optimize(error_value=None)); self.calls += 1
            if m.solver.status != "optimal": raise RuntimeError(m.solver.status)
            return value
        a = min(1., max(0., optimize(self.alpha, "max")))
        self.faces["maintenance"].lb = a
        self.faces["maintenance"].ub = a
        t = min(1., max(0., optimize(self.theta, "max")))
        self.faces["work"].lb = t
        self.faces["work"].ub = t
        oxygen = -self.ex["oxygen"].flux_expression
        permitted = None
        if interval:
            permitted = [max(0., optimize(oxygen, direction)) for direction in ["min", "max"]]
        order = ["glucose", "oxygen"] if self.spec["policy"] == "glucose_sparing" else ["oxygen", "glucose"]
        for role in order:
            value = max(0., optimize(-self.ex[role].flux_expression, "min"))
            self.faces[role].lb = max(0., value-1e-10)
            self.faces[role].ub = value+1e-10
        optimize(self.L1, "min")
        flux = np.array([r.flux for r in m.reactions])
        violation = max(max(r.lower_bound-v, v-r.upper_bound, 0.) for r, v in zip(m.reactions, flux))
        residue = float(np.abs(self.S@flux).max())
        assert max(violation, residue) < 1e-7
        assert max(abs(self.alpha.primal-a), abs(self.theta.primal-t)) < 1e-7, (self.alpha.primal, a, self.theta.primal, t)
        f = dict(zip(self.ids, flux))
        energy = 2*f["PAID_GLYCOLYSIS"]+self.spec["ATP_per_glucose_oxidative"]*f["PAID_OXIDATION"]
        atp_cost = sum(-r.metabolites.get(m.metabolites.HC00012_c, 0.)*f[r.id]
            for r in m.reactions if r.id not in ["PAID_GLYCOLYSIS", "PAID_OXIDATION"])
        assert abs(energy-atp_cost) < 1e-7
        return dict(alpha=float(self.alpha.primal), theta=float(self.theta.primal), flux=flux,
            net={role:f[rid] for role, rid in ROLES.items()}, max_Sv=residue, bound_error=violation,
            ATP_paid_pmol_h=energy, ATP_cost_pmol_h=atp_cost, oxygen_interval=permitted)
