"""Construct and replay a constant-rate witness, not a fitted metabolic policy.

An alternative feasible schedule can disprove necessity under the same stocks;
it cannot establish that real cells use this schedule or prove a global optimum.
"""
import json
import logging

import cobra
import numpy as np
import pandas as pd

from run_coculture import (HERE,ROOT,RedoxJoint,previous,build,ROLES,ARMS,
                          SER_TASK,C1_TASK,sha)


def initial_stock(spec,fields):
    stock=fields.initial_pmol.to_numpy().copy()
    idx={name:i for i,name in enumerate(fields.substrate)}
    for role,key in [('serine','ser_uM'),('glycine','gly_uM'),('formate','formate_uM')]:
        stock[idx[role]]=spec[key]*.02
    return stock,idx


def main():
    logging.getLogger('cobra').setLevel(logging.ERROR); cobra.Configuration().solver='glpk'
    protocol=json.loads((HERE/'protocol.json').read_text())
    fields=pd.read_csv(HERE/'fields.csv').fillna({'cobra_exchange':''})
    native=cobra.io.read_sbml_model(str(previous.NATIVE)); template=build(native)
    q=json.loads(previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    specs=[s for s in protocol['cases'] if s['family']=='core' and s['feed_rate']==0
           and s['arm'] in ['closed','funded']]
    reports=[]
    for spec in specs:
        joint=RedoxJoint(spec,fields,template,native,q)
        initial,idx=initial_stock(spec,fields)
        total=initial.copy()
        # At most 3 pmol/h O2, starting with 4 pmol and finite donor 36 pmol.
        # A stricter constant ceiling simplifies replay feasibility.
        total[idx['oxygen']]=min(initial[idx['oxygen']]+36.,3.*spec['horizon'])
        solved=joint.solve(total,spec['horizon'])
        stock=initial.copy(); external=np.zeros(len(stock)); donor=36.
        trace=[]; max_pool_error=0.; max_cn_error=0.
        for step in range(round(spec['horizon']/spec['dt'])):
            gross=np.zeros(len(fields))
            for j,row in enumerate(fields.itertuples()):
                for g in joint.ids:
                    rid=ROLES.get(row.substrate) if g=='tumor' else row.cobra_exchange
                    if rid in joint.r[g]:
                        v=joint.r[g][rid].flux*joint.factor[g]
                        gross[j]+=max(0.,-v)
            allowed=stock/spec['dt']; allowed[idx['serine']]+=solved['export']
            error=float(np.maximum(0.,gross-allowed).max()); max_pool_error=max(max_pool_error,error)
            raw=stock+spec['dt']*(solved['hnet']+solved['tnet'])
            assert raw.min()>-1e-7 and error<1e-7
            # Keep only numerical roundoff explicit; no new nutrient is supplied.
            fix=np.maximum(0.,-raw)
            assert fix.max()<1e-10
            stock=raw+fix
            feed=min(max(0.,4.-stock[idx['oxygen']]),3.*spec['dt'],donor)
            stock[idx['oxygen']]+=feed; external[idx['oxygen']]+=feed; donor-=feed
            time=(step+1)*spec['dt']
            residual=(stock-initial-external)@fields[['carbon_atoms','nitrogen_atoms']].to_numpy()
            residual+=time*solved['theta']*np.array([3*SER_TASK+C1_TASK,SER_TASK])
            max_cn_error=max(max_cn_error,float(abs(residual).max()))
            assert max_cn_error<1e-7
            trace.append(dict(step=step,time_h=time,serine_pmol=stock[idx['serine']],
                glycine_pmol=stock[idx['glycine']],formate_pmol=stock[idx['formate']],
                oxygen_pmol=stock[idx['oxygen']],oxygen_feed_pmol=feed,pool_error=error,
                carbon_residue=residual[0],nitrogen_residue=residual[1]))
        folder=HERE/'constant_witnesses'/spec['case']; folder.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(trace).to_csv(folder/'trajectory.csv',index=False)
        vectors=[dict(group=g,reaction=rid,flux=float(v),lb=joint.r[g][rid].lower_bound,
                      ub=joint.r[g][rid].upper_bound) for g in joint.ids
                 for rid,v in zip(joint.ids[g],solved['vectors'][g])]
        pd.DataFrame(vectors).to_csv(folder/'constant_fluxes.csv',index=False)
        original=json.loads((HERE/'outputs'/spec['case']/'summary.json').read_text())
        row=dict(case=spec['case'],tumor_variant=spec['tumor_variant'],capacity_fraction=spec['capacity_fraction'],
            arm=spec['arm'],interval_selected_task_h=original['task_h'],constant_feasible_task_h=solved['theta']*spec['horizon'],
            hepatic_maintenance=solved['hep'],tumor_maintenance=solved['maintenance'],
            export_pmol=solved['export']*spec['horizon'],max_pool_error=max_pool_error,
            max_CN_error=max_cn_error,initial_stocks_and_feed_rule_unchanged=True,
            scope='Feasible witness replayed, not a calibrated policy or proof of global optimality')
        reports.append(row)
        print(spec['case'],'stepwise',round(row['interval_selected_task_h'],6),
              'constant witness',round(row['constant_feasible_task_h'],6),flush=True)
    pd.DataFrame(reports).to_csv(HERE/'constant_protocol_comparison.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in protocol['protected_files'].items())
    (HERE/'constant_witness_checks.json').write_text(json.dumps(dict(cases=len(reports),all_replays_passed=True,
        protected_inputs_unchanged=True,not_global_optimization=True),indent=2))


if __name__=='__main__': main()
