"""Energy-only biosynthetic burden diagnostic; no complete protein biomass."""
import hashlib
import json
import logging
from pathlib import Path
import sys

import cobra
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'notebooks/serine_reference_independent_demands_2026_10_23'))
from run_reference import IndependentJoint,engine

BUDGETS={'operational':.0105,'Ser_residue_floor':.056*4.27,
         'Ser_fraction_0.10':.056*4.27/.10,'Ser_fraction_0.05':.056*4.27/.05}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class EnergyJoint(IndependentJoint):
    def __init__(self,spec,fields,template,native,q):
        super().__init__(spec,fields,template,native,q)
        # Replace, rather than add to, the old work demand to avoid double counting.
        self.model.solver.constraints['paid_energy_requirement'].set_linear_coefficients(
            {self.theta:-spec['work_ATP_at_full_service_pmol_h']})


def main():
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver='glpk'
    reference=ROOT/'notebooks/serine_reference_independent_demands_2026_10_23'
    previous=json.loads((reference/'reference_contract.json').read_text())
    base=next(s for s in previous['cases'] if s['family']=='primary' and s['variant']=='curated_transport'
              and s['tumor_variant']=='cytosolic_PPP' and s['gly_uM']==100.)
    specs=[]
    for target in [8.071428571428573,12.]:
        for arm in ['closed','funded']:
            for budget,work in BUDGETS.items():
                specs.append(base|dict(case=f'energy_{len(specs):03d}',arm=arm,
                    target_service_h=target,budget=budget,work_ATP_at_full_service_pmol_h=work))
    inputs=[reference/'run_reference.py',ROOT/'notebooks/serine_paid_redox_2026_10_22/redox_model.py',
        ROOT/'notebooks/serine_resource_paid_energy_2026_10_22/paid_energy.py',
        ROOT/'notebooks/serine_redox_coculture_2026_10_22/fields.csv',
        ROOT/'config/PhysiCell_settings.xml',ROOT/'config/hepatonet_pan_zonal_official_fastcore_reduced.xml',
        ROOT/'custom_modules/custom.cpp',ROOT/'addons/dFBA/src/dfba_intracellular.cpp',
        ROOT/'config/hepatonet_original_13_08.xml']
    protected={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    protocol=dict(cases=specs,declared_before_execution=True,protected=protected,
        reference_per_residue_ATP_equivalent=4.27,
        source='https://doi.org/10.1186/2049-3002-1-20',
        transfer='generic NCI60 modeling cost, not measured PATU demand',
        budgets=BUDGETS,unknown_Ser_residue_fractions=[.10,.05],
        scope='Energy-only burden. Other amino-acid material not supplied/consumed as biomass. No calibrated translation or growth',
        energy_equivalence='ATP/GTP costs lumped as ATP equivalents, no explicit GTP regeneration model',
        preserved='Fixed C1, finite stocks, inherited transport/respiratory ceilings, same populations and medium')
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    fields=pd.read_csv(inputs[3]).fillna({'cobra_exchange':''})
    native=cobra.io.read_sbml_model(str(engine.previous.NATIVE)); template=engine.build(native)
    q=json.loads(engine.previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    rows=[]
    for spec in specs:
        joint=EnergyJoint(spec,fields,template,native,q)
        m,p=joint.model,joint.model.problem
        initial,idx=engine.initial_stock(spec,fields)
        total=initial.copy(); total[idx['oxygen']]=36.
        for name,amount in zip(fields.substrate,total):
            joint.budget[name].ub=max(0.,float(amount))/12.
        joint.ha.lb=joint.ha.ub=1.
        joint.ta.lb=joint.ta.ub=1.
        goal=spec['target_service_h']/12.
        joint.theta.lb=joint.theta.ub=goal
        z=p.Variable('TUMOR_COMMON_CAPACITY',lb=0,ub=1)
        m.add_cons_vars(z)
        for rid in ['EX_GLY_c','EX_FOR_c']:
            m.add_cons_vars(p.Constraint(-joint.r['tumor'][rid].flux_expression-.14*z,
                ub=0,name='burden_capacity_'+rid))
        for rid in ['r0199','r0479','r0343']:
            m.add_cons_vars(p.Constraint(joint.r['tumor'][rid].flux_expression-.14*z,
                ub=0,name='burden_capacity_'+rid))
        m.objective=p.Objective(z,direction='min')
        value=m.slim_optimize()
        fixed_status=m.solver.status
        minimum=float(value) if fixed_status=='optimal' else None
        if fixed_status=='optimal':
            z.lb=max(0.,minimum); z.ub=min(1.,max(0.,minimum)+1e-8)
        else:
            assert fixed_status=='infeasible',fixed_status
            # Recover the achievable service with full exploratory pathway capacity.
            joint.theta.lb=0.; joint.theta.ub=1.
            z.lb=z.ub=1.
        result=joint.solve(total,12.)
        f={rid:r.flux for rid,r in joint.r['tumor'].items()}
        assert abs(f['TASK_ATP']-.0035-spec['work_ATP_at_full_service_pmol_h']*result['theta'])<1e-7
        net=result['hnet']+result['tnet']; stock=initial.copy(); external=np.zeros(len(stock)); donor=36.
        gross=np.zeros(len(fields))
        for j,row in enumerate(fields.itertuples()):
            for g in joint.r:
                rid=engine.ROLES.get(row.substrate) if g=='tumor' else row.cobra_exchange
                if rid in joint.r[g]:
                    gross[j]+=max(0.,-joint.factor[g]*joint.r[g][rid].flux)
        trace=[]
        for k in range(120):
            allowed=stock/.1; allowed[idx['serine']]+=result['export']
            pe=float(np.maximum(0,gross-allowed).max())
            stock+=.1*net
            feed=min(max(0.,4.-stock[idx['oxygen']]),.3,donor)
            donor-=feed; stock[idx['oxygen']]+=feed; external[idx['oxygen']]+=feed
            t=(k+1)*.1
            inv=stock-initial-external-t*net
            cn=(stock-initial-external)@fields[['carbon_atoms','nitrogen_atoms']].to_numpy()
            cn+=t*np.array([3*f['DM_SER_c']+f['DM_BOOKKEEP_C1'],f['DM_SER_c']])
            assert stock.min()>-1e-9 and max(pe,abs(inv).max(),abs(cn).max())<1e-7
            trace.append(dict(time_h=t,Ser_pmol=stock[idx['serine']],pool_error=pe,
                CN_error=float(abs(cn).max()),inventory_error=float(abs(inv).max())))
        out=HERE/'outputs'/spec['case']; out.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(trace).to_csv(out/'trajectory.csv',index=False)
        pd.DataFrame([dict(group=g,reaction=rid,flux=r.flux,lb=r.lower_bound,ub=r.upper_bound)
            for g in joint.r for rid,r in joint.r[g].items()]).to_csv(out/'fluxes.csv',index=False)
        # This analytic energy bound follows only from the inherited glucose and O2 uptake ceilings.
        glc_cap=.14; o2_cap=.14
        ideal_energy=2*glc_cap+(24.-2.)*min(glc_cap,o2_cap/6.)
        energy_relaxed_theta=(ideal_energy-.0035)/spec['work_ATP_at_full_service_pmol_h']
        rows.append(spec|dict(fixed_target_status=fixed_status,minimum_capacity_fraction=minimum,
            achieved_service_h=12*result['theta'],export_pmol=12*result['export'],
            net_Ser_assembly_pmol=12*(f['r0230']+f['CAP_SER_m_c']),
            ATP_total_pmol=12*result['atp'],ATP_maintenance_pmol=.0035*12,
            ATP_work_pmol=12*spec['work_ATP_at_full_service_pmol_h']*result['theta'],
            ATP_formate_pmol=12*f['r0228'],ATP_PPP_pmol=12*f['r0353'],
            glucose_pmol=-12*f['EX_E_GLC'],oxygen_pmol=-12*f['EX_E_O2'],
            ideal_energy_ceiling_pmol_h=ideal_energy,
            relaxed_service_upper_bound_h=12*min(1.,energy_relaxed_theta),
            max_CN_error=max(r['CN_error'] for r in trace),max_inventory_error=max(r['inventory_error'] for r in trace)))
        assert abs(rows[-1]['ATP_total_pmol']-sum(rows[-1][key] for key in
            ['ATP_maintenance_pmol','ATP_work_pmol','ATP_formate_pmol','ATP_PPP_pmol']))<1e-7
        assert result['theta']<=min(1,energy_relaxed_theta)+1e-7
        pd.DataFrame(rows).to_csv(HERE/'energy_budget_comparison.csv',index=False)
        print(spec['case'],spec['budget'],fixed_status,round(rows[-1]['achieved_service_h'],5),flush=True)
    df=pd.DataFrame(rows)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    verification=dict(all_passed=True,cases=len(df),target_feasible=int(df.fixed_target_status.eq('optimal').sum()),
        target_infeasible=int(df.fixed_target_status.eq('infeasible').sum()),replayed_intervals=len(df)*120,
        max_CN_error=float(df.max_CN_error.max()),max_inventory_error=float(df.max_inventory_error.max()),
        ATP_partition_checked=True,relaxed_energy_bound_checked=True,protected_inputs_unchanged=True)
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')


if __name__=='__main__':
    main()
