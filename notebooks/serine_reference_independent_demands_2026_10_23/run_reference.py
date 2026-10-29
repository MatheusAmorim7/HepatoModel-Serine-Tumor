"""Paired finite-stock source interventions with independent fixed C1 demand."""
import hashlib
import json
import logging
from pathlib import Path
import sys

import cobra
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'notebooks/serine_publication_reference_2026_10_22'))
import advance_reference as engine


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class IndependentJoint(engine.RedoxJoint):
    def __init__(self,spec,fields,template,native,q):
        super().__init__(spec,fields,template,native,q)
        con=self.model.solver.constraints['minimum_DM_BOOKKEEP_C1']
        con.set_linear_coefficients({self.theta:0.})
        con.ub=spec['fixed_C1_pmol_h']
        con.lb=spec['fixed_C1_pmol_h']


def plan():
    old=json.loads((ROOT/'notebooks/serine_redox_coculture_2026_10_22/protocol.json').read_text())
    base=next(s for s in old['cases'] if s['case']=='direct_only_C0_Q0_closed')
    cases=[]
    for hepatic in ['native_C1','curated_transport']:
        for tumor in ['direct_only','cytosolic_PPP']:
            for gly,formate in [(0.,0.),(100.,500.)]:
                for arm in ['closed','funded','PHGDH','PHGDH_SHMT']:
                    cases.append(base|dict(case=f'reference_{len(cases):03d}',
                        variant=hepatic,tumor_variant=tumor,capacity_fraction=0. if tumor=='direct_only' else 1.,
                        arm=arm,ser_uM=25.,gly_uM=gly,formate_uM=formate,
                        fixed_C1_pmol_h=.028,family='primary'))
    for tumor in ['direct_only','cytosolic_PPP']:
        for arm in ['closed','funded','PHGDH','PHGDH_SHMT']:
            cases.append(base|dict(case=f'reference_{len(cases):03d}',variant='curated_transport',
                tumor_variant=tumor,capacity_fraction=0. if tumor=='direct_only' else 1.,
                arm=arm,ser_uM=100.,gly_uM=0.,formate_uM=0.,fixed_C1_pmol_h=.028,family='Ser_rescue'))
    # Source priority is prescribed; check the alternate priority for the principal alternative.
    primary=[s for s in cases if s['family']=='primary' and s['variant']=='curated_transport'
             and s['tumor_variant']=='cytosolic_PPP']
    for s in primary:
        cases.append(s|dict(case=f'reference_{len(cases):03d}',priority='task_first',family='priority_check'))
    assert len(cases)==48
    return cases


def execute(spec,fields,template,native,q):
    joint=IndependentJoint(spec,fields,template,native,q)
    initial,idx=engine.initial_stock(spec,fields)
    total=initial.copy()
    total[idx['oxygen']]=min(initial[idx['oxygen']]+36.,3.*spec['horizon'])
    try:
        result=joint.solve(total,spec['horizon'])
    except cobra.exceptions.Infeasible:
        return spec|dict(status='infeasible_fixed_C1')
    flux={rid:r.flux for rid,r in joint.r['tumor'].items()}
    ser_rate=flux['DM_SER_c']; c1_rate=flux['DM_BOOKKEEP_C1']
    assert abs(ser_rate-engine.SER_TASK*result['theta'])<1e-7
    assert abs(c1_rate-spec['fixed_C1_pmol_h'])<1e-7
    assert abs(flux['TASK_ATP']-.0035*result['maintenance']-.0105*result['theta'])<1e-7
    net=result['hnet']+result['tnet']
    stock=initial.copy(); external=np.zeros(len(stock)); donor=36.
    gross=np.zeros(len(fields))
    for j,row in enumerate(fields.itertuples()):
        for g in joint.r:
            rid=engine.ROLES.get(row.substrate) if g=='tumor' else row.cobra_exchange
            if rid in joint.r[g]:
                gross[j]+=max(0.,-joint.factor[g]*joint.r[g][rid].flux)
    trace=[]
    for k in range(round(spec['horizon']/spec['dt'])):
        allowed=stock/spec['dt']; allowed[idx['serine']]+=result['export']
        poolerror=float(np.maximum(0,gross-allowed).max())
        stock=stock+spec['dt']*net
        assert stock.min()>-1e-9 and poolerror<1e-7
        feed=min(max(0,4.-stock[idx['oxygen']]),3.*spec['dt'],donor)
        donor-=feed; external[idx['oxygen']]+=feed; stock[idx['oxygen']]+=feed
        t=(k+1)*spec['dt']
        residue=stock-initial-external-t*net
        cn=(stock-initial-external)@fields[['carbon_atoms','nitrogen_atoms']].to_numpy()
        cn+=t*np.array([3*ser_rate+c1_rate,ser_rate])
        assert max(abs(residue).max(),abs(cn).max())<1e-7
        trace.append(dict(time_h=t,Ser_pmol=stock[idx['serine']],Gly_pmol=stock[idx['glycine']],
            formate_pmol=stock[idx['formate']],oxygen_pmol=stock[idx['oxygen']],
            inventory_error=float(abs(residue).max()),CN_error=float(abs(cn).max()),pool_error=poolerror))
    out=HERE/'outputs'/spec['case']; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(trace).to_csv(out/'trajectory.csv',index=False)
    pd.DataFrame([dict(group=g,reaction=rid,flux=r.flux,lb=r.lower_bound,ub=r.upper_bound)
        for g in joint.r for rid,r in joint.r[g].items()]).to_csv(out/'fluxes.csv',index=False)
    horizon=spec['horizon']
    return spec|dict(status='optimal',Ser_service_h=horizon*result['theta'],
        Ser_withdrawn_pmol=horizon*ser_rate,C1_delivered_pmol=horizon*c1_rate,
        export_pmol=horizon*result['export'],hep_maintenance=result['hep'],tumor_maintenance=result['maintenance'],
        net_tumor_Ser_assembly_pmol=horizon*(flux['r0230']+flux['CAP_SER_m_c']),
        tumor_ATP_pmol=horizon*result['atp'],tumor_Glc_pmol=-horizon*flux['EX_E_GLC'],
        tumor_O2_pmol=-horizon*flux['EX_E_O2'],O2_delivered_pmol=external[idx['oxygen']],
        max_CN_error=max(r['CN_error'] for r in trace),
        max_inventory_error=max(r['inventory_error'] for r in trace),
        max_pool_error=max(r['pool_error'] for r in trace))


def main():
    logging.getLogger('cobra').setLevel(logging.ERROR)
    cobra.Configuration().solver='glpk'
    protected={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'config/PhysiCell_settings.xml',
        ROOT/'config/hepatonet_pan_zonal_official_fastcore_reduced.xml',
        ROOT/'custom_modules/custom.cpp',ROOT/'addons/dFBA/src/dfba_intracellular.cpp',
        ROOT/'notebooks/serine_paid_redox_2026_10_22/redox_model.py',
        ROOT/'notebooks/serine_resource_paid_energy_2026_10_22/paid_energy.py',
        Path(engine.__file__)]}
    cases=plan()
    contract=dict(cases=cases,declared_before_execution=True,protected=protected,
        population_equivalents=dict(basal_hepatocytes=252,local_hepatocytes=12,tumor=7),
        volume_nL=20,horizon_h=12,dt_h=.1,
        independent_outputs=dict(Ser_rate_at_full_service_pmol_h=.056,C1_fixed_pmol_h=.028),
        energy=dict(maintenance_pmol_ATP_h=.0035,work_at_full_Ser_service_pmol_ATP_h=.0105,
            oxidative_ATP_per_glucose=24,status='operational chemistry, NOT full protein translation cost'),
        policy='Hepatic and tumor maintenance first; source/task priority recorded; glucose then oxygen then L1',
        budgets='Finite common stocks; same O2 donor/feedback; actual O2 delivery tracked',
        source='Prescribed potential export ceiling; no tumor induction calibration',
        no_new_rate_calibrated=True,full_biomass_or_growth=False,
        scope='Conditional chemical reference; tests remove demand coupling, not experimental validation')
    (HERE/'reference_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    fields=pd.read_csv(ROOT/'notebooks/serine_redox_coculture_2026_10_22/fields.csv').fillna({'cobra_exchange':''})
    native=cobra.io.read_sbml_model(str(engine.previous.NATIVE)); template=engine.build(native)
    q=json.loads(engine.previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    rows=[]
    for i,spec in enumerate(cases):
        result=execute(spec,fields,template,native,q)
        rows.append(result)
        pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
        print(i+1,len(cases),result['status'],result.get('Ser_service_h'),flush=True)
    df=pd.DataFrame(rows); feasible=df[df.status.eq('optimal')]
    assert feasible[['hep_maintenance','tumor_maintenance']].min().min()>=1-1e-7
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    (HERE/'verification.json').write_text(json.dumps(dict(all_passed=True,cases=len(df),
        optimal=len(feasible),infeasible_fixed_C1=int(df.status.ne('optimal').sum()),
        replayed_intervals=len(feasible)*120,
        max_CN_error=float(feasible.max_CN_error.max()),
        max_inventory_error=float(feasible.max_inventory_error.max()),
        protected_inputs_unchanged=True),indent=2)+'\n')


if __name__=='__main__':
    main()
