"""Restrict only local hepatic Gly uptake; pair external Ser with achieved export."""
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
sys.path.insert(0,str(ROOT/'notebooks/serine_biosynthetic_energy_2026_10_23'))
from run_energy import EnergyJoint,engine,BUDGETS


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def setup(spec,fields,template,native,q):
    joint=EnergyJoint(spec,fields,template,native,q)
    initial,idx=engine.initial_stock(spec,fields)
    total=initial.copy();total[idx['oxygen']]=36.
    total[idx['serine']]+=12*spec.get('external_rate_pmol_h',0.)
    for name,amount in zip(fields.substrate,total):
        joint.budget[name].ub=max(0.,float(amount))/12.
    joint.ha.lb=joint.ha.ub=1.;joint.ta.lb=joint.ta.ub=1.
    joint.r['tumor']['EX_E_GLC'].bounds=(-.5,0.)
    joint.r['tumor']['EX_E_O2'].bounds=(-3.,0.)
    gly=joint.r['local']['EX_HC00045_s']
    gly.lower_bound=-spec['local_Gly_ceiling_pmol_h']/joint.factor['local']
    return joint,initial,idx,total


def threshold(base,fields,template,native,q):
    rows=[]
    for arm in ['funded','PHGDH']:
        spec=base|dict(arm=arm,local_Gly_ceiling_pmol_h=1.)
        joint,initial,idx,total=setup(spec,fields,template,native,q)
        m,p=joint.model,joint.model.problem
        joint.theta.lb=joint.theta.ub=0.
        ser=joint.r['local'][engine.previous.hepatic.SER]
        ser.lower_bound=ser.upper_bound
        uptake=-joint.factor['local']*joint.r['local']['EX_HC00045_s'].flux_expression
        cap=p.Variable('LOCAL_GLY_MINIMUM',lb=0.)
        m.add_cons_vars([cap,p.Constraint(uptake-cap,ub=0.,name='local_gly_min')])
        m.objective=p.Objective(cap,direction='min')
        value=max(0.,float(m.slim_optimize(error_value=None)))
        below='zero_minimum'
        if value>1e-5:
            joint.r['local']['EX_HC00045_s'].lower_bound=-(value-1e-5)/joint.factor['local']
            m.slim_optimize();below=m.solver.status
            assert below=='infeasible'
        rows.append(dict(arm=arm,target_export_pmol_h=.024,minimum_local_Gly_pmol_h=value,
            Gly_per_Ser=value/.024,below_minimum_status=below,theta_for_threshold=0.,C1_fixed_pmol_h=.028))
    pd.DataFrame(rows).to_csv(HERE/'minimum_local_Gly.csv',index=False)
    critical=next(r['minimum_local_Gly_pmol_h'] for r in rows if r['arm']=='PHGDH')
    assert abs(critical-.048)<1e-7
    return critical


def execute(spec,fields,template,native,q):
    joint,initial,idx,total=setup(spec,fields,template,native,q)
    try:
        result=joint.solve(total,12.)
    except cobra.exceptions.Infeasible:
        return spec|dict(status='infeasible_fixed_C1_maintenance')
    f={rid:r.flux for rid,r in joint.r['tumor'].items()}
    rate=spec.get('external_rate_pmol_h',0.)
    gross=np.zeros(len(fields))
    for j,row in enumerate(fields.itertuples()):
        for g in joint.r:
            rid=engine.ROLES.get(row.substrate) if g=='tumor' else row.cobra_exchange
            if rid in joint.r[g]:
                gross[j]+=max(0.,-joint.factor[g]*joint.r[g][rid].flux)
    stock=initial.copy();external=np.zeros(len(stock));Odonor=36.;Sdonor=12*rate
    net=result['hnet']+result['tnet'];trace=[]
    for k in range(120):
        feed=min(.1*rate,Sdonor);Sdonor-=feed
        stock[idx['serine']]+=feed;external[idx['serine']]+=feed
        allowed=stock/.1;allowed[idx['serine']]+=result['export']
        pe=float(np.maximum(0.,gross-allowed).max());stock+=.1*net
        oxy=min(max(0.,4.-stock[idx['oxygen']]),.3,Odonor)
        Odonor-=oxy;stock[idx['oxygen']]+=oxy;external[idx['oxygen']]+=oxy
        t=(k+1)*.1
        inv=stock-initial-external-t*net
        cn=(stock-initial-external)@fields[['carbon_atoms','nitrogen_atoms']].to_numpy()
        cn+=t*np.array([3*f['DM_SER_c']+f['DM_BOOKKEEP_C1'],f['DM_SER_c']])
        assert stock.min()>-1e-9 and max(pe,abs(inv).max(),abs(cn).max())<1e-7
        trace.append(dict(time_h=t,Ser_pmol=stock[idx['serine']],
            cumulative_hepatic_Ser_pmol=t*result['export'],cumulative_external_Ser_pmol=external[idx['serine']],
            CN_error=float(abs(cn).max()),inventory_error=float(abs(inv).max()),pool_error=pe))
    out=HERE/'outputs'/spec['case'];out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(trace).to_csv(out/'trajectory.csv',index=False)
    pd.DataFrame([dict(group=g,reaction=rid,flux=r.flux,lb=r.lower_bound,ub=r.upper_bound,
        group_factor=joint.factor[g]) for g in joint.r for rid,r in joint.r[g].items()]).to_csv(out/'fluxes.csv',index=False)
    local_gly=max(0.,-joint.factor['local']*joint.r['local']['EX_HC00045_s'].flux)
    assert local_gly<=spec['local_Gly_ceiling_pmol_h']+1e-7
    row=spec|dict(status='optimal',Ser_service_h=12*result['theta'],hepatic_export_pmol=12*result['export'],
        external_Ser_pmol=external[idx['serine']],local_Gly_uptake_pmol=12*local_gly,
        tumor_net_Ser_assembly_pmol=12*(f['r0230']+f['CAP_SER_m_c']),
        tumor_ATP_pmol=12*result['atp'],max_CN_error=max(t['CN_error'] for t in trace),
        max_inventory_error=max(t['inventory_error'] for t in trace))
    for group,vector in [('hepatic',result['hnet']),('tumor',result['tnet'])]:
        for role in ['glucose','oxygen','glycine','formate','glutamine','ammonia']:
            row[group+'_'+role+'_net_pmol']=12*vector[idx[role]]
    return row


def main():
    logging.getLogger('cobra').setLevel(logging.ERROR);cobra.Configuration().solver='glpk'
    prior=json.loads((ROOT/'notebooks/serine_dose_matched_support_2026_10_23/protocol.json').read_text())
    base=next(s for s in prior['cases'] if s['context']=='burden10_G0.5' and s['intervention']=='hepatic_source'
              and s['policy']=='glucose_sparing')
    paths=[ROOT/'config/PhysiCell_settings.xml',ROOT/'config/hepatonet_pan_zonal_official_fastcore_reduced.xml',
        ROOT/'custom_modules/custom.cpp',ROOT/'addons/dFBA/src/dfba_intracellular.cpp',
        ROOT/'notebooks/serine_biosynthetic_energy_2026_10_23/run_energy.py',
        ROOT/'notebooks/serine_reference_independent_demands_2026_10_23/run_reference.py',
        ROOT/'notebooks/serine_redox_coculture_2026_10_22/fields.csv',ROOT/'config/hepatonet_original_13_08.xml']
    protected={str(p.relative_to(ROOT)):sha(p) for p in paths}
    declaration=dict(declared_before_execution=True,protected=protected,
        scope='Local Gly uptake capacity; external Gly concentration and tumor capacities unchanged',
        context='Inherited energy burden fSer0.10, tumor glucose cap0.5, C1fixed0.028, 12h,20nL,264heps+7tumor',
        design='PHGDH vs source intact, closed export, external full target dose, external achieved-dose matched',
        cap_fractions_of_computed_threshold=[0.,.5,1.,2.],tumor_variants=['direct_only','cytosolic_PPP'],
        policy='glucose_sparing main; oxygen_sparing checks at zero and threshold',
        matching='external achieved rate equals PHGDH hepatic export found at the same local cap and tumor variant',
        requirement='Maintenance fixed, C1 fixed; source and Ser service optimized, no forced infeasible export')
    (HERE/'protocol.json').write_text(json.dumps(declaration,indent=2)+'\n')
    fields=pd.read_csv(paths[6]).fillna({'cobra_exchange':''})
    native=cobra.io.read_sbml_model(str(engine.previous.NATIVE));template=engine.build(native)
    q=json.loads(engine.previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    critical=threshold(base,fields,template,native,q)
    rows=[];case_number=0
    for tumor in ['direct_only','cytosolic_PPP']:
        for fraction in [0.,.5,1.,2.]:
            cap=critical*fraction
            for policy in ['glucose_sparing']+(['oxygen_sparing'] if fraction in [0.,1.] else []):
                achieved=None
                interventions=['hepatic_PHGDH','external_actual'] if policy=='oxygen_sparing' else [
                    'closed','hepatic_source','hepatic_PHGDH','external_reference','external_actual']
                for intervention in interventions:
                    rate=.024 if intervention=='external_reference' else (achieved if intervention=='external_actual' else 0.)
                    assert rate is not None
                    spec=base|dict(case=f'localcap_{case_number:03d}',policy=policy,
                        tumor_variant=tumor,capacity_fraction=0. if tumor=='direct_only' else 1.,
                        intervention=intervention,arm={'hepatic_source':'funded','hepatic_PHGDH':'PHGDH'}.get(intervention,'closed'),
                        local_Gly_ceiling_pmol_h=cap,threshold_fraction=fraction,external_rate_pmol_h=rate)
                    row=execute(spec,fields,template,native,q);rows.append(row);case_number+=1
                    if intervention=='hepatic_PHGDH':
                        assert row['status']=='optimal';achieved=row['hepatic_export_pmol']/12.
                    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
                    print(case_number,tumor,fraction,policy,intervention,row['status'],row.get('Ser_service_h'),flush=True)
    df=pd.DataFrame(rows);good=df[df.status.eq('optimal')]
    assert len(df)==48
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    (HERE/'verification.json').write_text(json.dumps(dict(all_passed=True,cases=len(df),optimal=len(good),
        replay_intervals=len(good)*120,computed_local_threshold_pmol_h=critical,
        max_CN_error=float(good.max_CN_error.max()),max_inventory_error=float(good.max_inventory_error.max()),
        protected_inputs_unchanged=True),indent=2)+'\n')


if __name__=='__main__':
    main()
