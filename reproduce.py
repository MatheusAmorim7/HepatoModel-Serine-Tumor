"""Replay the original poster protocols with the original model code and GLPK."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'notebooks/serine_supply_salvage_2026_10_26'))
import run_supply as supply
import run_context_anchors as context

TOL=1e-7

def check_sources():
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name

def select_cases(full=False):
    demand=supply.demand.plan()
    offer=supply.plan()
    anchors=context.plan()
    if not full:
        demand=[r for r in demand if r['energy_mode']=='fixed_ATP'
                and r['composition'] in ['S0.5_P1_C1','S1_P1_C1','S1_P1_C2']
                and r['target'] in [None,.55]]
        offer=[r for r in offer if r['panel']=='analytical_boundary' or
               (r['external_Ser_pmol']==.5 and r['target'] in [None,.55] and
                ((r['HPRT_cap_pmol_h'] in [0.,.14]) or r['HX_pmol']==.2))]
        anchors=[r for r in anchors if r['HPRT_cap_pmol_h']==.14
                 and r['HX_pmol'] in [0.,.1] and r['target'] in [None,.55]]
    return [('demand',r) for r in demand]+[('supply',r) for r in offer]+[('context',r) for r in anchors]

def prepared(stage,row,cache,fields):
    if stage=='demand':
        joint,initial,idx,_=supply.demand.clone(cache[(row['salvage_condition'],row['channel'])],fields)
        supply.demand.configure(joint,row)
        return joint,initial,idx
    return supply.prepare(row,cache,fields)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='Replay all 2,200 archived scenarios')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    out=args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is not empty. Choose a new --out-dir; previous results are preserved.')
    out.mkdir(parents=True,exist_ok=True)
    check_sources()
    references={name:pd.read_csv(ROOT/'reference_results'/f'{name}_results.csv').set_index('case')
                for name in ['demand','supply','context']}
    inputs=supply.demand.load_inputs()
    fields=inputs[-1]
    cache=supply.demand.templates(inputs)
    cases=select_cases(args.full)
    rows=[]
    inventory=[]
    rechecks=[]
    metrics=['theta','hepatic_Ser_export_pmol','tumor_Ser_material_pmol','tumor_IMP_material_pmol',
             'tumor_C1_material_pmol','IMP_de_novo_pmol','IMP_salvage_pmol']
    for i,(stage,row) in enumerate(cases):
        joint,initial,idx=prepared(stage,row,cache,fields)
        theta,flux=supply.demand.solve(joint,row['target'] is not None)
        observed=row|dict(stage=stage)
        if flux is None:
            observed['status']='infeasible_fixed_target' if row['target'] is not None else 'infeasible_maintenance'
        else:
            validate=supply.demand.validate if stage=='demand' else supply.validate
            values,stocks=validate(joint,row,fields,initial,idx,theta,flux)
            observed|=dict(status='optimal')|values
            inventory.extend(dict(stage=stage,case=row['case'])|s for s in stocks)
        ref=references[stage].loc[row['case']]
        assert observed['status']==ref.status,(stage,row['case'],observed['status'],ref.status)
        if observed['status']=='optimal':
            for key in metrics:
                assert abs(observed[key]-ref[key])<TOL,(stage,row['case'],key,observed[key],ref[key])
        # A fresh model checks the raw objective without the later selected faces.
        fresh,_,_=prepared(stage,row,cache,fields)
        model=fresh.model
        source=fresh.factor['local']*fresh.r['local'][supply.demand.prior.community.shared.SER].flux_expression
        model.objective=model.problem.Objective(fresh.theta if row['target'] is None else source,
                                                direction='max' if row['target'] is None else 'min')
        raw=model.slim_optimize()
        if flux is None:
            assert model.solver.status=='infeasible'
        else:
            expected=theta if row['target'] is None else observed['hepatic_Ser_export_pmol']/12.
            assert model.solver.status=='optimal' and abs(raw-expected)<TOL
        rechecks.append(dict(stage=stage,case=row['case'],raw_objective_check='passed'))
        rows.append(observed)
        if (i+1)%20==0 or i+1==len(cases):
            print(f'{i+1}/{len(cases)} cases checked',flush=True)
    data=pd.DataFrame(rows)
    data.to_csv(out/'results.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'stocks.csv',index=False)
    pd.DataFrame(rechecks).to_csv(out/'objective_checks.csv',index=False)
    pairs=[]
    for stage,row in cases:
        if row['target'] is None or row['channel']!='serine_only':
            continue
        result=data[(data.stage==stage)&(data.case==row['case'])].iloc[0]
        pairs.append(dict(stage=stage,case=row['case'],target=row['target'],
            composition=row['composition'],salvage_condition=row.get('salvage_condition',''),
            external_Ser_pmol=row.get('external_Ser_pmol',.5),HX_pmol=row['HX_pmol'],
            HPRT_cap_pmol_h=row.get('HPRT_cap_pmol_h',.14 if row.get('HPRT')=='available' else 0.),
            status=result.status,hepatic_Ser_export_pmol=result.get('hepatic_Ser_export_pmol',np.nan)))
    pd.DataFrame(pairs).to_csv(out/'source_requirements.csv',index=False)
    from plot_results import make_figures
    make_figures(data,out)
    check_sources()
    checks=dict(cases=len(data),scope='full' if args.full else 'poster',
        status_counts=data.status.value_counts().to_dict(),
        original_reference_parity=True,raw_objectives_rechecked=True,
        intracellular_mass_balance=True,finite_stock_replays=True,
        carbon_nitrogen_and_ATP_checks=True,source_files_unchanged=True,
        growth_inferred=False,experimental_validation_claimed=False)
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    print('Results:',out,flush=True)

if __name__=='__main__':
    main()
