"""Restore explicit directions from HepatoNet1 Supplementary Data 1.2.

Check stoichiometric orientation before translating arrows into bounds.
No reaction deletions and no empirical correction based on a desired flux.
"""
import csv
import json
import logging
import re
from pathlib import Path
import cobra
import pandas as pd
from cobra.flux_analysis import pfba
from run_comparison import HERE, ROOT, ATP, SER, O2, GLC, load, apply, sha

SOURCE = HERE/'sources/44320_2010_BFMSB201062_MOESM4_ESM.txt'

def side(text):
    result = {}
    for token in text.split(' + '):
        match = re.fullmatch(r'(?:(\d+(?:\.\d+)?)\s+)?(HC\d+_[a-z])',token.strip())
        if not match:
            raise ValueError('Unparsed stoichiometry: '+token)
        coef, mid = match.groups()
        result[mid] = result.get(mid,0)+float(coef or 1)
    return result

def apply_published(m):
    rows = []
    with SOURCE.open() as handle:
        records = list(csv.reader(handle,delimiter='\t'))
    for fields in records[1:]:
        if len(fields)<2 or fields[0] not in m.reactions:
            continue
        rid, equation = fields[:2]
        parts = re.split(r' (<=>|-->|<--) ',equation)
        if len(parts)!=3:
            raise ValueError(equation)
        left, arrow, right = parts
        lhs,rhs = side(left),side(right)
        vector = {mid:rhs.get(mid,0)-lhs.get(mid,0) for mid in lhs.keys()|rhs.keys()}
        vector = {mid:v for mid,v in vector.items() if v}
        r = m.reactions.get_by_id(rid)
        actual = {met.id:coef for met,coef in r.metabolites.items()}
        if vector == actual:
            sign = 1
        elif {mid:-v for mid,v in vector.items()} == actual:
            sign = -1
        else:
            raise ValueError('Stoichiometry mismatch '+rid)
        old = r.bounds
        bounds = {'<=>':(-1000,1000),'-->':(0,1000),'<--':(-1000,0)}[arrow]
        if sign == -1:
            bounds = (-bounds[1],-bounds[0])
        r.bounds = bounds
        rows.append(dict(reaction=rid,equation=equation,orientation=sign,
                         old_lb=old[0],old_ub=old[1],lb=bounds[0],ub=bounds[1],changed=old!=bounds))
    return rows

def main():
    logging.getLogger('cobra').setLevel(logging.ERROR)
    out = HERE/'direction_curation'
    out.mkdir(exist_ok=True)
    m,info = load('full_native')
    rows = apply_published(m)
    pd.DataFrame(rows).to_csv(out/'published_directions.csv',index=False)
    m.id = 'hepatonet_full_supplement_directions_diagnostic'
    m.notes['scope'] = 'Direction-only diagnostic. Fails closed-medium ATP control. Not ready for PhysiCell. Loopless constraints are not included in SBML.'
    cobra.io.write_sbml_model(m,str(out/'hepatonet_full_direction_candidate.xml'))
    results, vectors = [], []
    for medium in ['closed','no_organic','glucose_glutamine','candidate_AA']:
        for arm in ['none','PHGDH','PHGDH_SHMT']:
            with m:
                if medium == 'closed':
                    for r in m.boundary:
                        r.bounds = (0,0)
                else:
                    apply(m,medium,arm)
                for task in ['ATP_capacity','serine_capacity']:
                    with m:
                        m.reactions.get_by_id(ATP).bounds = (0,1000) if task=='ATP_capacity' else (5,5)
                        m.objective = ATP if task=='ATP_capacity' else SER
                        m.objective_direction = 'max'
                        sol = m.optimize()
                        tag=dict(medium=medium,arm=arm,task=task,status=sol.status,value=None)
                        if sol.status=='optimal':
                            tag['value']=float(sol.objective_value)
                            witness=pfba(m)
                            vectors += [tag|dict(reaction=rid,flux=float(v)) for rid,v in witness.fluxes.items() if abs(v)>1e-8]
                        results.append(tag)
                        print(tag,flush=True)
        pd.DataFrame(results).to_csv(out/'controls.csv',index=False)
    pd.DataFrame(vectors).to_csv(out/'active_fluxes.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(SOURCE),
        matched_reactions=len(rows),changed_reactions=sum(r['changed'] for r in rows),
        input=info,scope='Published arrow directions only. Gibbs/concentration constraints not implemented.',
        source_model_unchanged=sha(ROOT/info['input'])==info['sha256']),indent=2))

if __name__=='__main__':
    main()
