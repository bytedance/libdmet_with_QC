# Copyright (c) Bytedance Inc. 
# SPDX-License-Identifier: GPL-3.0-Only

"""
Useful functions for magnetic order guesses.
"""

import numpy as np
import numpy
from pyscf import gto, lib
from pyscf.pbc import gto as pbcgto
 
THRESH_HOLD = 0.001
def init_guess_by_minao(mol):
    '''Generate initial guess density matrix based on ANO basis, then project
    the density matrix to the basis set defined by ``mol``

    Returns:
        Density matrix, 2D ndarray

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> scf.hf.init_guess_by_minao(mol)
    array([[ 0.94758917,  0.09227308],
           [ 0.09227308,  0.94758917]])
    '''
    from pyscf.scf import atom_hf
    from pyscf.scf import addons

    def minao_basis(symb, nelec_ecp):
        occ = []
        basis_ano = []
        if gto.is_ghost_atom(symb):
            return occ, basis_ano

        stdsymb = gto.mole._std_symbol(symb)
        basis_add = gto.basis.load('ano', stdsymb)
# coreshl defines the core shells to be removed in the initial guess
        coreshl = gto.ecp.core_configuration(nelec_ecp)
        #coreshl = (0,0,0,0)  # it keeps all core electrons in the initial guess
        for l in range(4):
            ndocc, frac = atom_hf.frac_occ(stdsymb, l)
            assert ndocc >= coreshl[l]
            degen = l * 2 + 1
            occ_l = [2,]*(ndocc-coreshl[l]) + [frac,]
            occ.append(numpy.repeat(occ_l, degen))
            basis_ano.append([l] + [b[:1] + b[1+coreshl[l]:ndocc+2]
                                    for b in basis_add[l][1:]])
        occ = numpy.hstack(occ)

        if nelec_ecp > 0:
            if symb in mol._basis:
                input_basis = mol._basis[symb]
            elif stdsymb in mol._basis:
                input_basis = mol._basis[stdsymb]
            else:
                raise KeyError(symb)

            basis4ecp = [[] for i in range(4)]
            for bas in input_basis:
                l = bas[0]
                if l < 4:
                    basis4ecp[l].append(bas)

            occ4ecp = []
            for l in range(4):
                nbas_l = sum((len(bas[1]) - 1) for bas in basis4ecp[l])
                ndocc, frac = atom_hf.frac_occ(stdsymb, l)
                ndocc -= coreshl[l]
                assert ndocc <= nbas_l

                occ_l = numpy.zeros(nbas_l)
                occ_l[:ndocc] = 2
                if frac > 0:
                    occ_l[ndocc] = frac
                occ4ecp.append(numpy.repeat(occ_l, l * 2 + 1))

            occ4ecp = numpy.hstack(occ4ecp)
            basis4ecp = lib.flatten(basis4ecp)

# Compared to ANO valence basis, to check whether the ECP basis set has
# reasonable AO-character contraction.  The ANO valence AO should have
# significant overlap to ECP basis if the ECP basis has AO-character.
            atm1 = gto.Mole()
            atm2 = gto.Mole()
            atom = [[symb, (0.,0.,0.)]]
            atm1._atm, atm1._bas, atm1._env = atm1.make_env(atom, {symb:basis4ecp}, [])
            atm2._atm, atm2._bas, atm2._env = atm2.make_env(atom, {symb:basis_ano}, [])
            atm1._built = True
            atm2._built = True
            s12 = gto.intor_cross('int1e_ovlp', atm1, atm2)
            if abs(numpy.linalg.det(s12[occ4ecp>0][:,occ>0])) > THRESH_HOLD:
                occ, basis_ano = occ4ecp, basis4ecp
            else:
                print(mol, 'Density of valence part of ANO basis '
                             'will be used as initial guess for %s', symb)
        return occ, basis_ano

    # Issue 548
    if any(gto.charge(mol.atom_symbol(ia)) > 96 for ia in range(mol.natm)):
        print(mol, 'MINAO initial guess is not available for super-heavy '
                    'elements. "atom" initial guess is used.')
        return init_guess_by_atom(mol)

    nelec_ecp_dic = dict([(mol.atom_symbol(ia), mol.atom_nelec_core(ia))
                          for ia in range(mol.natm)])

    basis = {}
    occdic = {}
    for symb, nelec_ecp in nelec_ecp_dic.items():
        occ_add, basis_add = minao_basis(symb, nelec_ecp)
        occdic[symb] = occ_add
        basis[symb] = basis_add

    occ = []
    new_atom = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if not gto.is_ghost_atom(symb):
            occ.append(occdic[symb])
            new_atom.append(mol._atom[ia])
    occ = numpy.hstack(occ)

    pmol = gto.Mole()
    pmol._atm, pmol._bas, pmol._env = pmol.make_env(new_atom, basis, [])
    pmol._built = True
    dm = addons.project_dm_nr2nr(pmol, numpy.diag(occ), mol)
# normalize eletron number
#    s = mol.intor_symmetric('int1e_ovlp')
#    dm *= mol.nelectron / (dm*s).sum()
    return dm
   
def guess_magmom(kmf, dm0=None, magmom="Ni:3d:2 Ni:3d:-2", normalized=True):
    """
    Make a guess from a magnetic order.
    The number of sites for an element needs to the same as the number of that element in the structure
    Attributes:
        cell    : cell object
        kmf     : initial k-sampling kmf object,
        magmom  : desire magnetic pattern    
    Returns:
        dm_kpts : (nkpts, nao, nao) ndarray
    """
    cell = kmf.cell
    sites = magmom.split()
    num_sites = len(sites)

    # Non-magnetic guess
    if dm0 is None:
        if hasattr(kmf, 'kpts'):
            nkpts = len(kmf.kpts)
        elif hasattr(kmf, 'kpt'):
            nkpts = len(kmf.kpt) 
        #nkpts = len(kmf.kpts)
        dm0 = init_guess_by_minao(cell)
        dm0 = np.asarray([dm0]*nkpts)*0.5
       
    dm_a = dm0.copy()
    dm_b = dm0.copy()
    index = []
    for i, site in enumerate(sites):
        element, orb, mag = site.split(':')
        site_str = '{} {} {}'.format(i,element,orb)
        idx = [i for i, s in enumerate(cell.ao_labels()) if site_str in s]
        dm_a[np.ix_(np.arange(nkpts),idx,idx)] += float(mag)/len(idx)/2
        dm_b[np.ix_(np.arange(nkpts),idx,idx)] -= float(mag)/len(idx)/2
        index += idx
       
    dm = np.asarray([dm_a,dm_b])
   
    # Normalize 1-RDM
    if normalized:
        print('The 1-RDM is normalized to the correct number of electron in kmf object')
        nao = cell.nao
        mask = np.zeros([2,nkpts,nao,nao])
        mask[np.ix_([0,1],np.arange(nkpts),index,index)] = 1
        s = kmf.get_ovlp(cell)
        ne = np.einsum('xkij,kji->x', dm, s).real
        nelec = np.asarray(kmf.nelec)
        diff = nelec - ne
        ne_sub = np.einsum('xkij,kji->x', dm[np.ix_([0,1],np.arange(nkpts),index,index)], s[np.ix_(np.arange(nkpts),index,index)]).real
        dm[0] = dm[0] + mask[0]*diff[0]/len(index)/nkpts
        dm[1] = dm[1] + mask[1]*diff[1]/len(index)/nkpts
   
    return dm
   
def guess_from_gamma(chkfile, kmf, normalized=True, symmetrized_threshold=None):
    """
    Take RDM-1 from a Gamma-point calculation and turns it in to k-sampling one
    """
    cell = kmf.cell
    nkpts = len(kmf.kpts)
    scfdat = lib.chkfile.load(chkfile, 'scf')
    mo_coeff = np.asarray(scfdat['mo_coeff'])
    mo_occ = np.asarray(scfdat['mo_occ'])
    if np.asarray(mo_occ).ndim == 4:
        mo_coeff_kpts = np.repeat(mo_coeff[:,np.newaxis,:,:], nkpts, axis=1)
        mo_occ_kpts = np.repeat(mo_occ[:,np.newaxis,:], nkpts, axis=1)
    else:
        mo_coeff_kpts = np.repeat(mo_coeff[:,:,:,:], nkpts, axis=1)
        mo_occ_kpts = np.repeat(mo_occ[:,:,:], nkpts, axis=1)
    dm = kmf.make_rdm1(mo_coeff_kpts=mo_coeff_kpts, mo_occ_kpts=mo_occ_kpts)
   
    # Symmetrize the spin
    if symmetrized_threshold is not None:
        mask_sym = np.int32(np.abs(dm[0] -dm[1]) < symmetrized_threshold)
        mask_asym = np.int32((mask_sym == 1) == False)
        dm_a_old = dm[0].copy()
        dm_b_old = dm[1].copy()
        dm[0] = mask_sym * (dm_a_old + dm_b_old)/2. + dm_a_old*mask_asym
        dm[1] = mask_sym * (dm_a_old + dm_b_old)/2. + dm_b_old*mask_asym
       
    # Normalize 1-RDM
    if normalized:
        print('The 1-RDM is normalized to the correct number of electron in kmf object')
        s = kmf.get_ovlp(cell)
        ne = np.einsum('xkij,kji->x', dm, s).real
        nelec = np.asarray(kmf.nelec)
        dm *= (nelec / ne).reshape(2,-1,1,1)

    return dm
   
def guess_flip(chkfile, kmf, magmom="Ni:3d:0 Ni:3d:1", normalized=True, symmetrized_threshold=None):
    """
    Make a guess by flipping the magnetic order from a converged RDM-1
    The number of sites for an element needs to the same as the number of that element in the structure
    Attributes:
        chkfile : checkpoint file name
        kmf     : initial k-sampling kmf object,
        magmom  : used to specify which atom and whether to flip it (1) or not (0).
    Returns:
        dm_kpts : (nkpts, nao, nao) ndarray
    """
    sites = magmom.split()
    num_sites = len(sites)
    cell = kmf.cell
   
    nkpts = len(kmf.kpts)
    scfdat = lib.chkfile.load(chkfile, 'scf')
    mo_coeff_kpts = np.asarray(scfdat['mo_coeff'])
    mo_occ_kpts = np.asarray(scfdat['mo_occ'])
    dm0 = kmf.make_rdm1(mo_coeff_kpts=mo_coeff_kpts, mo_occ_kpts=mo_occ_kpts)

    dm_a = dm0[0]
    dm_b = dm0[1]
    index = []
    for i, site in enumerate(sites):
        element, orb, flip = site.split(':')
        site_str = '{} {} {}'.format(i,element,orb)
        idx = [i for i, s in enumerate(cell.ao_labels()) if site_str in s]
        if flip =='1':
            dm_a_old = dm_a[np.ix_(np.arange(nkpts),idx,idx)].copy()
            dm_b_old = dm_b[np.ix_(np.arange(nkpts),idx,idx)].copy()
            dm_a[np.ix_(np.arange(nkpts),idx,idx)] = dm_b_old
            dm_b[np.ix_(np.arange(nkpts),idx,idx)] = dm_a_old
        index += idx
           
    dm = np.asarray([dm_a,dm_b])

    # Symmetrize the spin
    if symmetrized_threshold is not None:
        mask_sym = np.int32(np.abs(dm[0] -dm[1]) < symmetrized_threshold)
        mask_asym = np.int32((mask_sym == 1) == False)
        dm_a_old = dm[0].copy()
        dm_b_old = dm[1].copy()
        dm[0] = mask_sym * (dm_a_old + dm_b_old)/2. + dm_a_old*mask_asym
        dm[1] = mask_sym * (dm_a_old + dm_b_old)/2. + dm_b_old*mask_asym
       
    # Normalize 1-RDM
    if normalized:
        print('The 1-RDM is normalized to the correct number of electron in kmf object')
        nao = cell.nao
        mask = np.zeros([2,nkpts,nao,nao])
        mask[np.ix_([0,1],np.arange(nkpts),index,index)] = 1
        s = kmf.get_ovlp(cell)
        ne = np.einsum('xkij,kji->x', dm, s).real
        nelec = np.asarray(kmf.nelec)
        diff = nelec - ne
        ne_sub = np.einsum('xkij,kji->x', dm[np.ix_([0,1],np.arange(nkpts),index,index)], s[np.ix_(np.arange(nkpts),index,index)]).real
        dm[0] = dm[0] + mask[0]*diff[0]/len(index)/nkpts
        dm[1] = dm[1] + mask[1]*diff[1]/len(index)/nkpts
   
    return dm
   
def guess_from_cell1_to_cell2(cell1, chkfile, cell2, kpts):
    """
    Make a guess by projecting the RDM-1 of cell1 onto that of cell2
    Note only works for non-relatistic calculation
    Attributes:
        cell1, cell2    : two cell objects
        chkfile         : check file from the calculation on cell1
        kpts            : k-point list
    Returns:
        dm_kpts : (nkpts, nao, nao) ndarray
    """
    s22 = cell2.pbc_intor('int1e_ovlp', kpts=kpts)
    s21 = pbcgto.cell.intor_cross('int1e_ovlp', cell2, cell1, kpts=kpts)
   
    from pyscf.pbc.scf import kuhf
    scfdat = lib.chkfile.load(chkfile, 'scf')
    mo_coeff_kpts = np.asarray(scfdat['mo_coeff'])
    mo_occ_kpts = np.asarray(scfdat['mo_occ'])
    dm1  = kuhf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
   
    p21_kpts = []
    dm_kpts = []
    for i, kpt in enumerate(kpts):
        p21 = lib.cho_solve(s22[i], s21[i], strict_sym_pos=False)
        p21_kpts.append(p21)
        dm = lib.einsum('pi,nij,qj->npq', p21, dm1[:,i,:,:], p21.conj())
        dm_kpts.append(dm)
    dm_kpts = np.transpose(dm_kpts, [1,0,2,3])
    return dm_kpts

def mulliken_pop(cell, kmf, dm=None, s=None):
    '''Mulliken population analysis
    '''
    if s is None: s = kmf.get_ovlp(cell)
    if dm is None: dm = kmf.make_rdm1()

    nkpts = dm[0].shape[0]
    pop_a = 1/nkpts * np.einsum('kij,kji->i', dm[0], s).real
    pop_b = 1/nkpts * np.einsum('kij,kji->i', dm[1], s).real

    print(' ** Mulliken pop       alpha | beta **')
    for i, s in enumerate(cell.ao_labels()):
        print('pop of  %s %10.5f | %-10.5f' % (s, pop_a[i], pop_b[i]))

    print('In total          %10.5f | %-10.5f' % (sum(pop_a), sum(pop_b)))

    print(' ** Mulliken atomic charges   ( Nelec_alpha | Nelec_beta | Diff) **')
    nelec_a = np.zeros(cell.natm)
    nelec_b = np.zeros(cell.natm)
    for i, s in enumerate(cell.ao_labels(fmt=None)):
        nelec_a[s[0]] += pop_a[i]
        nelec_b[s[0]] += pop_b[i]
    chg = cell.atom_charges() - (nelec_a + nelec_b)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        print('charge of  %d%s =   %10.5f  (  %10.5f   %10.5f %10.5f )' %
                 (ia, symb, chg[ia], nelec_a[ia], nelec_b[ia], nelec_a[ia] - nelec_b[ia]))
    print('Total charge   =   %10.5f  (  %10.5f   %10.5f %10.5f )' % (sum(chg), sum(nelec_a),
                        sum(nelec_b), sum(nelec_a) - sum(nelec_b)))
    return (pop_a,pop_b), chg
   
LINDEP_THRESH = 1e-6
def eig(h_kpts, s_kpts):
    nkpts = len(h_kpts)
    eig_kpts = []
    mo_coeff_kpts = []
    for k in range(nkpts):
        d, t = np.linalg.eigh(s_kpts[k])
        x = t[:,d>LINDEP_THRESH] / np.sqrt(d[d>LINDEP_THRESH])
        xhx = x.T.dot(h_kpts[k]).dot(x)
        e, c = np.linalg.eigh(xhx)
        c = np.dot(x, c)
        eig_kpts.append(e)
        mo_coeff_kpts.append(c)
    return eig_kpts, mo_coeff_kpts

def eig_kuhf(h_kpts, s_kpts):
    e_a, c_a = eig(h_kpts[0], s_kpts)
    e_b, c_b = eig(h_kpts[1], s_kpts)
    return (e_a,e_b), (c_a,c_b)
