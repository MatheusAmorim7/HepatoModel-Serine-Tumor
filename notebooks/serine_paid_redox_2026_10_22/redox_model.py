"""Resource-accounted Ser/Gly/C1 diagnostic, not a tumor reconstruction."""
from pathlib import Path
import sys

import cobra
import numpy as np
from cobra.util.array import create_stoichiometric_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'notebooks/serine_resource_paid_energy_2026_10_22'))
from paid_energy import (PaidSelector, build, configure, atoms as old_atoms,
                         add, ROLES as OLD_ROLES, ATOMS as OLD_ATOMS,
                         MAINTENANCE, WORK_ATP, SER_TASK, C1_TASK, BACKGROUND)

PPP = ['r0353', 'r0199', 'r0479', 'r0343']
VARIANTS = ['direct_only', 'cytosolic_PPP', 'cytosolic_formate', 'mitochondrial_GCS']
ROLES = OLD_ROLES | dict(ammonia='EX_NH3_m', pentose_carbon='BOOKKEEP_PENTOSE_OUT')
ATOMS = OLD_ATOMS | dict(ammonia=(0, 1), pentose_carbon=(5, 0))
EXTRA_CN = dict(Glucose=(6, 0), **{'Glucose-6P': (6, 0),
    'Glucono-1,5-lactone-6P': (6, 0), '6-Phospho-D-gluconate': (6, 0),
    'Ribulose-5P': (5, 0)})


def atoms(met):
    name = met.name.rsplit('(', 1)[0]
    if met.id == 'BOOKKEEP_PENTOSE_c':
        return np.array([5, 0])
    return np.array(EXTRA_CN[name]) if name in EXTRA_CN else old_atoms(met)


class RedoxSelector(PaidSelector):
    def __init__(self, template, native, spec, q):
        super().__init__(template, spec, q)
        m = self.model
        variant = spec['variant']
        assert variant in VARIANTS
        base_variant = dict(direct_only='uptake_only', cytosolic_PPP='cytosolic_C1',
                            cytosolic_formate='plus_formate_oxidation',
                            mitochondrial_GCS='plus_mito_GCS')[variant]
        configured = configure(template, base_variant,
            dict(serine_uM=100, glycine_uM=100, formate_uM=100))
        # Transfer only internal route availability, preserving paid task constraints.
        for rid in ['r0230', 'r0293', 'r0371', 'r0228', 'r0231', 'r0295',
                    'r0338', 'r0663', 'r0159', 'r0227', 'CAP_GLY_c_m', 'CAP_SER_m_c']:
            m.reactions.get_by_id(rid).bounds = tuple(v*.14 for v in configured.reactions.get_by_id(rid).bounds)
        # External Ser must still be usable for the same cytosolic C1 task.
        # No C1 source exists in the direct control when Ser uptake is closed.
        m.reactions.r0230.bounds = (-140., 140.)
        for rid in PPP:
            reaction = native.reactions.get_by_id(rid).copy()
            reaction.bounds = (0, .14 if variant == 'cytosolic_PPP' else 0)
            reaction.notes['scope'] = 'Native chemical equation, forward diagnostic direction; no measured PATU capacity'
            m.add_reactions([reaction])
        # Both ATP-producing routes and the PPP compete for the same glucose pool.
        glc = next(met.id for met in m.reactions.r0353.metabolites if met.name == 'Glucose(c)')
        pentose = next(met.id for met in m.reactions.r0343.metabolites if met.name == 'Ribulose-5P(c)')
        add(m, 'GLUCOSE_TO_PPP', {'E_GLC_c': -1, glc: 1})
        m.add_metabolites([cobra.Metabolite('BOOKKEEP_PENTOSE_c',
            name='Bookkeeping pentose carbon, not observed secretion', compartment='c')])
        add(m, 'PENTOSE_ACCOUNTING', {pentose: -1, 'HC00011_c': -1,
                                    'HC00019_c': 1, 'BOOKKEEP_PENTOSE_c': 1})
        add(m, 'BOOKKEEP_PENTOSE_OUT', {'BOOKKEEP_PENTOSE_c': -1})
        # Conservative respiratory disposal: no ATP credit for this NADH.
        add(m, 'PAID_NADH_OX_m', {'HC00014_m': -1, 'E_O2_c': -.5,
                                  'HC00013_m': 1, 'HC00011_m': 1})
        m.reactions.PAID_NADH_OX_m.upper_bound = .14 if variant == 'mitochondrial_GCS' else 0
        m.reactions.PAID_NADH_OX_m.notes['scope'] = (
            'Effective NADH + half O2 disposal, zero ATP credit; not a calibrated ETC or membrane-potential model')
        add(m, 'CO2_MATRIX_TO_CYTOSOL', {'HC00021_m': -1, 'HC00021_c': 1})
        m.reactions.CO2_MATRIX_TO_CYTOSOL.upper_bound = .14 if variant == 'mitochondrial_GCS' else 0
        m.reactions.EX_CO2_m.bounds = (0, 0)
        self.ex = {role: m.reactions.get_by_id(rid) for role, rid in ROLES.items()}
        self.caps.update(ammonia=0., pentose_carbon=0.)
        for rid in spec.get('blocks', []):
            m.reactions.get_by_id(rid).bounds = (0, 0)
        self.ids = m.reactions.list_attr('id')
        self.S = create_stoichiometric_matrix(m)
        self.L1 = sum(r.forward_variable + r.reverse_variable for r in m.reactions)
        m.notes['scope'] = 'Resource-accounted chemistry only; fixed seven tumor equivalents, no growth or hepatocytes'
        m.notes['pentose'] = 'One-pass oxidative PPP. Pentose carbon is explicitly retained in an accounting pool, not measured secretion or biomass.'
        m.notes['redox'] = 'Formal cofactor ports remain closed. PPP or formate oxidation pays NADPH; effective oxygen-linked reaction disposes matrix NADH.'
        for r in m.reactions:
            if not r.boundary:
                residue = sum((atoms(met)*v for met, v in r.metabolites.items()), start=np.zeros(2))
                assert max(abs(residue)) < 1e-10, (r.id, residue)

    def solve(self, stock, dt, interval=False):
        result = super().solve(stock, dt, interval)
        f = dict(zip(self.ids, result['flux']))
        result['net'] = {role:f[rid] for role,rid in ROLES.items()}
        # Native folate directions: r0293 generates NADPH forward, consumes it reverse.
        result['NADPH_residue'] = 2*f['r0199'] + f['r0293'] + f['r0227']
        result['NADH_m_residue'] = f['r0295'] - f['PAID_NADH_OX_m']
        result['PPP_glucose'] = f['GLUCOSE_TO_PPP']
        result['redox_oxygen'] = .5*f['PAID_NADH_OX_m']
        assert max(abs(result['NADPH_residue']), abs(result['NADH_m_residue'])) < 1e-7
        assert max(abs(f[rid]) for rid in BACKGROUND) < 1e-9
        return result
