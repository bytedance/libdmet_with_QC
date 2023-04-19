# Copyright (c) Bytedance Inc. 
# SPDX-License-Identifier: GPL-3.0-Only

"""
Useful functions for magnetic order guesses.
"""

import numpy as np
import numpy


def vcor_to_vcor_I(dmet, restricted, bogoliubov, nscsites, frags, I, vcor):
    ''' Tranform the global u to the fragment u'''
    idx_range = frags[I][0]  
    vcor_I = dmet.VcorLocal_new(restricted, bogoliubov, nscsites, idx_range=idx_range)
    vcor_I.update(np.ones(vcor_I.length()))
    mask = abs(vcor_I.value - 1) < 1.e-9
    if restricted:
        mask[0][np.tril_indices(mask[0].shape[0], -1)] = False
        vcor_I.update(vcor.value[0][mask[0]])   
    else:
        mask[0][np.tril_indices(mask[0].shape[0], -1)] = False
        mask[1][np.tril_indices(mask[1].shape[0], -1)] = False   
        vcor_I.update(vcor.value[mask])   
        
    return vcor_I

def vcor_I_to_vcor(dmet, restricted, bogoliubov, nscsites, frags, vcor, vcor_I_list):
    ''' Tranform the fragment u to the global u '''
    
    vcor_value_list = [vcor_I.value for vcor_I in vcor_I_list]
    vcor_value = np.asarray(vcor_value_list).sum(axis=0)
    
    vcor.update(np.ones(vcor.length()))
    mask = abs(vcor.value - 1) < 1.e-9
    if restricted:
        mask[0][np.tril_indices(mask[0].shape[0], -1)] = False
        vcor.update(vcor_value[0][mask[0]])
    else:
        mask[0][np.tril_indices(mask[0].shape[0], -1)] = False
        mask[1][np.tril_indices(mask[1].shape[0], -1)] = False
        vcor.update(vcor_value[mask])
    return vcor

def get_magmom(iao_site, frags, rhoImp_list):
    ''' Obtain the magnetic moment at site with its iao_site indices'''
    offset = 0
    charge_a = 0.
    charge_b = 0.
    for i, frag in enumerate(frags):
        frag_val_virt = sum(frag[:-1],[])
        rho_a, rho_b = rhoImp_list[i]
        rho_a = rho_a.diagonal()
        rho_b = rho_b.diagonal()
        for orb in range(len(frag_val_virt)):
            if frag_val_virt[orb] in iao_site:
                charge_a += rho_a[orb]
                charge_b += rho_b[orb]
    return charge_a - charge_b
    
    
