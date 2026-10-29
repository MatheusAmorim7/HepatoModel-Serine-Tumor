"""Separately declared nutrient anchors; not a reconstruction of Plasmax/DMEM."""
import csv
import gzip
import json

import pandas as pd

import run_supply as stage


def plan():
    rows=[]
    for ser in [.5,2.8,8.]:
        for hx in [0.,.004,.1]:
            for cap in [0.,.004,.14]:
                for channel in ['closed','serine_only']:
                    for target in [None,.55]:
                        rows.append(stage.COMPOSITION|dict(panel='nutrient_context',
                            external_Ser_pmol=ser,Ser_uM=ser*50,HX_pmol=hx,HX_uM=hx*50,
                            HPRT_cap_pmol_h=cap,channel=channel,target=target))
    assert len(rows)==108
    return [r|dict(case=f'context_{i:03d}') for i,r in enumerate(rows)]


def main():
    out=stage.HERE/'context_anchors';out.mkdir(exist_ok=True)
    assert not (out/'results.csv').exists()
    cases=plan();frozen=stage.protected()
    protocol=dict(cases=cases,declared_before_execution=True,
        script_sha256=stage.demand.sha(__file__),protected=frozen,
        reference='https://www.nature.com/articles/s41467-021-26395-5',
        nutrient_selection='Ser140uM and400uM correspond to described media conditions; HX5uM Plasmax and HX0.2uM low mouse context. Ser25uM remains the diagnostic reference.',
        scope='Only these nutrient concentrations are changed. Other medium, cells, times, demands and capacities do not recreate the published experiments. No validation or physiological threshold claim.',
        HPRT='All three pathway ceilings are exploratory; no enzyme/protein/drug calibration',
        independent_validation=False,active_PhysiCell_changed=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    inputs=stage.demand.load_inputs();fields=inputs[-1]
    cache=stage.demand.templates(inputs)
    results=[];rechecks=[]
    with gzip.open(out/'fluxes.csv.gz','wt',newline='') as vf,(out/'stocks.csv').open('w',newline='') as sf:
        fw=csv.DictWriter(vf,fieldnames=['case','group','reaction','flux']);fw.writeheader()
        iw=None
        for i,row in enumerate(cases):
            joint,initial,idx=stage.prepare(row,cache,fields)
            theta,flux=stage.demand.solve(joint,row['target'] is not None)
            if flux is None:
                results.append(row|dict(status='infeasible_fixed_target'))
            else:
                metrics,inventory=stage.validate(joint,row,fields,initial,idx,theta,flux)
                results.append(row|dict(status='optimal')|metrics)
                for g in flux:
                    fw.writerows(dict(case=row['case'],group=g,reaction=rid,flux=v) for rid,v in flux[g].items())
                if iw is None:
                    iw=csv.DictWriter(sf,fieldnames=['case']+list(inventory[0]));iw.writeheader()
                iw.writerows(dict(case=row['case'])|inv for inv in inventory)
            # New model, raw objective only: do not re-use selected optimal faces.
            direct,_,_=stage.prepare(row,cache,fields)
            m=direct.model
            expression=(direct.theta if row['target'] is None else direct.factor['local']*
                direct.r['local'][stage.demand.prior.community.shared.SER].flux_expression)
            m.objective=m.problem.Objective(expression,direction='max' if row['target'] is None else 'min')
            value=m.slim_optimize()
            if flux is None:
                assert m.solver.status=='infeasible'
            else:
                expected=theta if row['target'] is None else metrics['hepatic_Ser_export_pmol']/stage.HOURS
                assert m.solver.status=='optimal' and abs(float(value)-expected)<stage.TOL
            rechecks.append(dict(case=row['case'],check='passed'))
            if (i+1)%18==0:
                pd.DataFrame(results).to_csv(out/'results.csv',index=False)
                print('context',i+1,'/',len(cases),flush=True)
    data=pd.DataFrame(results).set_index('case',drop=False)
    saved_vectors=pd.read_csv(out/'fluxes.csv.gz').groupby('case',sort=False)
    saved_stocks=pd.read_csv(out/'stocks.csv').groupby('case',sort=False)
    for row in cases:
        r=data.loc[row['case']]
        if r.status!='optimal':
            continue
        joint,initial,idx=stage.prepare(row,cache,fields)
        v=saved_vectors.get_group(row['case'])
        flux={g:part.set_index('reaction').flux.to_dict() for g,part in v.groupby('group')}
        metrics,inventory=stage.validate(joint,row,fields,initial,idx,float(r.theta),flux)
        for key,value in metrics.items():
            assert abs(value-r[key])<stage.TOL
        saved=saved_stocks.get_group(row['case']).set_index('substrate')
        for inv in inventory:
            assert max(abs(v-saved.loc[inv['substrate'],k]) for k,v in inv.items() if k!='substrate')<stage.TOL
    assert all(stage.demand.sha(stage.ROOT/name)==h for name,h in frozen.items())
    pd.DataFrame(rechecks).to_csv(out/'objective_rechecks.csv',index=False)
    checks=dict(cases=len(cases),status_counts=data.status.value_counts().to_dict(),
        all_objectives_rechecked=True,saved_flux_stock_checks=True,protected_unchanged=True,
        biological_validation=False)
    (out/'audit_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
