"""Paired shared-stock diagnostic with paid tumor Gly/C1 alternatives."""
import argparse
import gzip
import hashlib
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
REDOX = ROOT/'notebooks/serine_paid_redox_2026_10_22'
PRIOR = ROOT/'notebooks/serine_external_supply_competition_2026_10_22'
sys.path.insert(0,str(REDOX))
from redox_model import RedoxSelector, ROLES, BACKGROUND, build, SER_TASK, C1_TASK, MAINTENANCE, WORK_ATP
sys.path.insert(0,str(ROOT/'notebooks/serine_paired_paid_coculture_2026_10_22'))
from run_paired import Joint, previous

ARMS = ['closed','funded','PHGDH','PHGDH_SHMT']
ROUTE_BLOCK = dict(cytosolic_PPP='r0199',cytosolic_formate='r0227',mitochondrial_GCS='r0295')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    base = dict(variant='curated_transport',policy='glucose_sparing',priority='source_first',
                medium='reference',control='none',dt=.1,horizon=12.,yield_ATP=24.,
                ser_uM=25.,gly_uM=100.,basal_uptake_ceiling=.084,blocks=[])
    rows=[]
    for tumor in ['direct_only','cytosolic_PPP','cytosolic_formate','mitochondrial_GCS']:
        for capacity in ([0.] if tumor=='direct_only' else [.1,1.]):
            for feed in [0.,.084]:
                for arm in ARMS:
                    # All arms/variants receive the same Gly+formate challenge;
                    # different allowed routes, not different media, are compared.
                    rows.append(base|dict(case=f'{tumor}_C{capacity:g}_Q{feed:g}_{arm}',
                        family='core',tumor_variant=tumor,capacity_fraction=capacity,
                        feed_rate=feed,donor_budget=feed*12,formate_uM=500.,arm=arm))
    for tumor in ROUTE_BLOCK:
        parent=next(s for s in rows if s['tumor_variant']==tumor and s['capacity_fraction']==1.
                    and s['feed_rate']==0 and s['arm']=='closed')
        for arm in ['closed','funded']:
            rows.append(parent|dict(case=tumor+'_tumor_block_'+arm,family='tumor_block',
                                    arm=arm,blocks=[ROUTE_BLOCK[tumor]]))
        rows.append(parent|dict(case=tumor+'_no_formate',family='supply_control',formate_uM=0.))
        rows.append(parent|dict(case=tumor+'_no_initial_glycine',family='supply_control',gly_uM=0.))
    for tumor in ['cytosolic_PPP','mitochondrial_GCS']:
        for arm in ARMS:
            parent=next(s for s in rows if s['family']=='core' and s['tumor_variant']==tumor
                        and s['capacity_fraction']==1. and s['feed_rate']==0 and s['arm']==arm)
            rows.append(parent|dict(case=tumor+'_task_first_'+arm,family='priority',priority='task_first'))
    for arm in ['closed','funded']:
        parent=next(s for s in rows if s['case']==f'cytosolic_PPP_C0.1_Q0_{arm}')
        rows.append(parent|dict(case='halfdt_'+arm,family='refinement',dt=.05,parent_case=parent['case']))
    for tumor in ['direct_only','cytosolic_PPP','mitochondrial_GCS']:
        parent=next(s for s in rows if s['tumor_variant']==tumor)
        rows.append(parent|dict(case=tumor+'_no_resources',family='negative',arm='funded',
                                control='no_resources',horizon=1.,feed_rate=0.,donor_budget=0.))
    return rows


def fields():
    f=pd.read_csv(PRIOR/'fields.csv')
    f=pd.concat([f,pd.DataFrame([dict(substrate='pentose_carbon',cobra_exchange='',
        carbon_atoms=5,nitrogen_atoms=0,initial_uM=0.,initial_pmol=0.,
        diffusion_status='accounting_only_not_a_spatial_field')])],ignore_index=True)
    f['cobra_exchange']=f.cobra_exchange.fillna('')
    assert f.substrate.is_unique and len(f)==22
    return f


class RedoxJoint(Joint):
    def __init__(self,spec,f,template,native,q):
        super().__init__(spec,f,template,q)
        paid=RedoxSelector(template,native,dict(variant=spec['tumor_variant'],
            ATP_per_glucose_oxidative=spec['yield_ATP'],policy=spec['policy'],blocks=spec['blocks']),q)
        m=self.model
        cap=spec['capacity_fraction']
        for rid in ['r0199','r0479','r0343','r0227','r0295','PAID_NADH_OX_m',
                    'CAP_GLY_c_m','CAP_SER_m_c']:
            r=paid.model.reactions.get_by_id(rid)
            r.bounds=tuple(v*cap for v in r.bounds)
        for r in paid.model.reactions:
            if r.id not in m.reactions:
                m.add_reactions([r.copy()])
            else:
                m.reactions.get_by_id(r.id).bounds=r.bounds
        self.groups['tumor']=paid.model.copy()
        self.r['tumor']={r.id:m.reactions.get_by_id(r.id) for r in paid.model.reactions}
        self.tumor_caps=paid.caps.copy()
        for role in ['glycine','formate']:
            self.tumor_caps[role]*=cap
        for role,rid in ROLES.items():
            self.r['tumor'][rid].bounds=(-self.tumor_caps[role],0 if role in ['glucose','oxygen'] else 140.)
        self.r['basal'][previous.hepatic.SER].bounds=(-spec['basal_uptake_ceiling']/self.factor['basal'],0.)
        # Replace every shared-pool expression after adding tumor reactions.
        m.remove_cons_vars(list(self.budget.values()))
        self.gross,self.net,self.budget={},{},{}
        p=m.problem
        for row in f.itertuples():
            gross,net=0,0
            for g in self.r:
                rid=ROLES.get(row.substrate) if g=='tumor' else row.cobra_exchange
                if rid in self.r[g]:
                    r=self.r[g][rid]
                    gross+=self.factor[g]*r.reverse_variable
                    net+=self.factor[g]*r.flux_expression
            self.gross[row.substrate],self.net[row.substrate]=gross,net
            expr=gross-self.export if row.substrate=='serine' else gross
            self.budget[row.substrate]=p.Constraint(expr,ub=0,name='redox_shared_'+row.substrate)
            m.add_cons_vars(self.budget[row.substrate])
        self.ids={g:list(self.r[g]) for g in self.r}
        self.S={g:create_stoichiometric_matrix(original) for g,original in self.groups.items()}
        self.l1={var:self.factor[g] for g in self.r for r in self.r[g].values()
                 for var in [r.forward_variable,r.reverse_variable]}

    def solve(self,stock,dt):
        m,p=self.model,self.model.problem
        for face in self.faces.values():
            face.lb=face.ub=None
        for row,amount in zip(self.fields.itertuples(),stock):
            self.budget[row.substrate].ub=max(0.,float(amount))/dt
        def opt(expr,direction):
            m.objective=p.Objective(expr,direction=direction)
            self.calls+=1
            return float(m.slim_optimize(error_value=None))
        hierarchy=['hep','tumor_maintenance']+(['export','tasks'] if self.spec['priority']=='source_first' else ['tasks','export'])
        expressions=dict(hep=self.ha,tumor_maintenance=self.ta,export=self.export,tasks=self.theta)
        for key in hierarchy:
            value=opt(expressions[key],'max')
            self.faces[key].lb=self.faces[key].ub=value
        for key in (['glucose','oxygen'] if self.spec['policy']=='glucose_sparing' else ['oxygen','glucose']):
            value=max(0.,opt(self.gross[key],'min'))
            self.faces[key].lb=max(0.,value-1e-10)
            self.faces[key].ub=value+1e-10
        m.objective=p.Objective(0,direction='min')
        m.objective.set_linear_coefficients(self.l1)
        m.slim_optimize(error_value=None)
        self.calls+=1
        vectors={g:np.array([self.r[g][rid].flux for rid in self.ids[g]]) for g in self.r}
        residual=max(float(abs(self.S[g]@v).max()) for g,v in vectors.items())
        violation=max(max(self.r[g][rid].lower_bound-v,v-self.r[g][rid].upper_bound,0.)
                      for g in self.r for rid,v in zip(self.ids[g],vectors[g]))
        hnet,tnet,gross=[np.zeros(len(self.fields)) for _ in range(3)]
        for j,row in enumerate(self.fields.itertuples()):
            for g in self.r:
                rid=ROLES.get(row.substrate) if g=='tumor' else row.cobra_exchange
                if rid in self.r[g]:
                    v=self.r[g][rid].flux*self.factor[g]
                    (tnet if g=='tumor' else hnet)[j]+=v
                    gross[j]+=max(0.,-v)
        export=self.factor['local']*self.r['local'][previous.hepatic.SER].flux
        limits=stock/dt
        limits[self.fields.index[self.fields.substrate.eq('serine')][0]]+=export
        pool_error=float(np.maximum(0,gross-limits).max())
        t={rid:r.flux for rid,r in self.r['tumor'].items()}
        atp=2*t['PAID_GLYCOLYSIS']+self.spec['yield_ATP']*t['PAID_OXIDATION']
        atp_error=abs(atp-t['TASK_ATP']-t['r0228']-t['r0353'])
        redox_error=max(abs(2*t['r0199']+t['r0293']+t['r0227']),abs(t['r0295']-t['PAID_NADH_OX_m']))
        assert max(residual,violation,pool_error,atp_error,redox_error)<1e-7
        assert max(abs(t[rid]) for rid in BACKGROUND)<1e-9
        return dict(hep=float(self.ha.primal),maintenance=float(self.ta.primal),theta=float(self.theta.primal),
            export=export,basal_uptake=max(0.,-self.factor['basal']*self.r['basal'][previous.hepatic.SER].flux),
            hnet=hnet,tnet=tnet,vectors=vectors,residual=residual,bound_error=violation,pool_error=pool_error,
            atp=atp,atp_error=atp_error,redox_error=redox_error,PPP_glucose=t['GLUCOSE_TO_PPP'],
            redox_oxygen=.5*t['PAID_NADH_OX_m'])


def simulate(spec,f,template,native,q):
    out=HERE/'outputs'/spec['case']; out.mkdir(parents=True,exist_ok=True)
    joint=RedoxJoint(spec,f,template,native,q)
    stock=f.initial_pmol.to_numpy().copy()
    idx={name:i for i,name in enumerate(f.substrate)}
    for role,key in [('serine','ser_uM'),('glycine','gly_uM'),('formate','formate_uM')]:
        stock[idx[role]]=spec[key]*.02
    if spec['control']=='no_resources':
        stock[:]=0.
    initial=stock.copy()
    hcum,tcum,external,correction=[np.zeros(len(f)) for _ in range(4)]
    ser_donor=spec['donor_budget']; oxy_donor=0. if spec['control']=='no_resources' else 36.
    totals=dict(task_h=0.,hep_h=0.,maintenance_h=0.,export_pmol=0.,basal_uptake_pmol=0.,
                PPP_glucose_pmol=0.,redox_oxygen_pmol=0.,ATP_paid_pmol=0.)
    eligible=True; eligible_h=eligible_tasks=0.
    trace,inventory=[],[]
    start=time.perf_counter()
    with gzip.open(out/'fluxes.csv.gz','wt') as stream:
        stream.write('step,'+','.join(g+'__'+rid for g in joint.ids for rid in joint.ids[g])+'\n')
        for step in range(round(spec['horizon']/spec['dt'])):
            before=stock.copy()
            feed=min(spec['feed_rate']*spec['dt'],max(0.,ser_donor))
            stock[idx['serine']]+=feed; external[idx['serine']]+=feed; ser_donor-=feed
            after_feed=stock.copy()
            result=joint.solve(stock,spec['dt'])
            stream.write(str(step)+','+','.join(str(float(v)) for g in joint.ids for v in result['vectors'][g])+'\n')
            raw=stock+spec['dt']*(result['hnet']+result['tnet'])
            assert raw.min()>-1e-7
            fix=np.maximum(0.,-raw); stock=raw+fix; correction+=fix
            hcum+=spec['dt']*result['hnet']; tcum+=spec['dt']*result['tnet']
            oxy_feed=min(max(0.,4.-stock[idx['oxygen']]),3.*spec['dt'],oxy_donor)
            stock[idx['oxygen']]+=oxy_feed; external[idx['oxygen']]+=oxy_feed; oxy_donor-=oxy_feed
            for key,result_key in [('task_h','theta'),('hep_h','hep'),('maintenance_h','maintenance'),
                ('export_pmol','export'),('basal_uptake_pmol','basal_uptake'),('PPP_glucose_pmol','PPP_glucose'),
                ('redox_oxygen_pmol','redox_oxygen'),('ATP_paid_pmol','atp')]:
                totals[key]+=result[result_key]*spec['dt']
            eligible=eligible and min(result['hep'],result['maintenance'])>=1.-1e-7
            if eligible:
                eligible_h+=spec['dt']; eligible_tasks+=result['theta']*spec['dt']
            balance=stock-initial-external-hcum-tcum-correction
            material=(stock-initial-external-correction)@f[['carbon_atoms','nitrogen_atoms']].to_numpy()
            material+=[3*SER_TASK*totals['task_h']+C1_TASK*totals['task_h'],SER_TASK*totals['task_h']]
            assert max(abs(balance).max(),abs(material).max())<1e-7
            trace.append(dict(step=step,time_h=(step+1)*spec['dt'],theta=result['theta'],hep=result['hep'],
                maintenance=result['maintenance'],export_pmol_h=result['export'],basal_uptake_pmol_h=result['basal_uptake'],
                max_Sv=result['residual'],bound_error=result['bound_error'],pool_error=result['pool_error'],
                ATP_error=result['atp_error'],redox_error=result['redox_error'],carbon_residue=material[0],
                nitrogen_residue=material[1],serine_feed_pmol=feed,oxygen_feed_pmol=oxy_feed,eligible=eligible))
            for j,row in enumerate(f.itertuples()):
                inventory.append(dict(step=step,substrate=row.substrate,initial=initial[j],before=before[j],
                    after_feed=after_feed[j],remaining=stock[j],external=external[j],hepatic=hcum[j],
                    tumor=tcum[j],correction=correction[j],residue=balance[j]))
    pd.DataFrame(trace).to_csv(out/'trajectory.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'inventory.csv',index=False)
    pd.DataFrame([dict(group=g,reaction=rid,lb=r.lower_bound,ub=r.upper_bound)
                  for g in joint.r for rid,r in joint.r[g].items()]).to_csv(out/'bounds.csv',index=False)
    row=spec|totals|dict(eligible_h=eligible_h,eligible_task_h=eligible_tasks,
        hepatic_net_serine_pmol=hcum[idx['serine']],tumor_serine_net_pmol=tcum[idx['serine']],
        serine_external_pmol=external[idx['serine']],oxygen_external_pmol=external[idx['oxygen']],
        seconds=time.perf_counter()-start,calls=joint.calls)
    (out/'summary.json').write_text(json.dumps(row,indent=2))
    print(spec['case'],'tasks',round(row['task_h'],6),'source',round(row['export_pmol'],6),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    logging.getLogger('cobra').setLevel(logging.ERROR); cobra.Configuration().solver='glpk'
    specs=plan()
    if args.smoke:
        specs=[s|dict(case='smoke_'+s['tumor_variant']+'_'+s['arm'],horizon=.2) for s in specs
               if s['family']=='core' and s['feed_rate']==0 and s['capacity_fraction']==(0. if s['tumor_variant']=='direct_only' else 1.)
               and s['arm'] in ['closed','funded']]
    assert len({s['case'] for s in specs})==len(specs)
    protected=json.loads((PRIOR/'protocol.json').read_text())['inputs_sha256'].copy()
    for folder in [PRIOR,REDOX,ROOT/'notebooks/serine_PATU8902_public_route_evidence']:
        for path in folder.glob('*'):
            if path.is_file(): protected[str(path.relative_to(ROOT))]=sha(path)
    f=fields(); f.to_csv(HERE/'fields.csv',index=False)
    protocol=dict(cases=specs,declared_before_execution=True,counts=previous.COUNTS,volume_nL=20.,
        capacity='0.1 and 1 times previous exploratory ceilings; no parameter range inferred from expression or CRISPR',
        medium='Same initial Ser 25 uM, Gly 100 uM, formate 500 uM for core comparisons; other inherited fields unchanged. Culture-inspired challenge, not in vivo or a reproduced assay',
        serum_feed='Fixed finite Ser doses, not perfusion. Same dose/timing within each pair. O2 target/ceiling identical, delivered O2 may differ and is tracked',
        priority='Hepatic maintenance and tumor maintenance first; compare source-first vs task-first. A centralized allocation rule, not measured cell behavior',
        basal='Uptake allowed but not required; demand remains uncalibrated',
        scope='Fixed populations, no growth/signaling/spatial inference. Explicit tumor redox, pentose carbon accounting, no full thermodynamic validation',
        protected_files=protected,inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [Path(__file__),REDOX/'redox_model.py',HERE/'fields.csv',ROOT/'notebooks/serine_paired_paid_coculture_2026_10_22/run_paired.py']})
    suffix='_smoke' if args.smoke else ''
    (HERE/('protocol'+suffix+'.json')).write_text(json.dumps(protocol,indent=2))
    native=cobra.io.read_sbml_model(str(previous.NATIVE)); template=build(native)
    q=json.loads(previous.PARKER.read_text())['positive_net_consumption_nmol_per_million_cells_h']
    rows=[]
    for spec in specs:
        saved=HERE/'outputs'/spec['case']/'summary.json'
        if args.resume and saved.exists():
            row=json.loads(saved.read_text()); assert all(row[k]==v for k,v in spec.items())
        else:
            row=simulate(spec,f,template,native,q)
        rows.append(row); pd.DataFrame(rows).to_csv(HERE/('summary'+suffix+'.csv'),index=False)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    (HERE/('execution'+suffix+'.json')).write_text(json.dumps(dict(completed=len(rows),inputs_unchanged=True)))


if __name__=='__main__': main()
