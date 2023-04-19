# Copyright (c) Bytedance Inc. 
# SPDX-License-Identifier: GPL-3.0-Only

"""
DMET for BN
"""
# Select different solvers on line 272-274
import os, sys
import numpy as np
import scipy.linalg as la

from pyscf import lib, fci, ao2mo
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df, dft, cc, tools

from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.basis_transform import eri_transform
from libdmet.lo.iao import reference_mol, get_labels, get_idx
from libdmet.basis_transform.make_basis import symmetrize_kmf
from libdmet.utils.misc import max_abs, mdot, kdot, read_poscar

import libdmet.utils.logger as log
import libdmet.dmet.Hubbard as dmet

from libdmet_qc.solver.yao_qc import QCyao
from libdmet_qc.utils import tool, dmet_tool

log.verbose = "RESULT"
np.set_printoptions(4, linewidth=1000, suppress=True)

max_memory = 110000
exxdiv = None

### ************************************************************
### Creating BN structure
### ************************************************************
def make_BN_cell(a=2.30):
    os.system("cp ./BN-P1 ./BN_{0:2.2f}".format(a))
    os.system("sed -i -e 's/2.5000000000/{0:2.10f}/g' ./BN_{1:2.2f}".format(a, a))
    os.system("sed -i -e 's/-1.2500000000/-{0:2.10f}/g' ./BN_{1:2.2f}".format(a/2, a))
    os.system("sed -i -e 's/2.1650635095/{0:2.10f}/g' ./BN_{1:2.2f}".format(a*np.cos(np.pi/6), a))
    cell = read_poscar("./BN_{0:2.2f}".format(a))
    return cell

### ************************************************************
### System settings
### ************************************************************

dis = [2.30+i*0.02 for i in range(0,5)]
kpoint = 5
sc_energy_ls = []
os_energy_ls = []


for a in dis:

    cell = make_BN_cell(a=a)
    cell.basis   = 'gth-dzvp-molopt-sr'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 5
    cell.precision = 1e-12
    cell.spin = 0
    cell.max_memory = 100000
    cell.build()


    kmesh = [kpoint, kpoint, 1]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    exxdiv = None

    minao = 'gth-szv-molopt-sr'
    kmf_conv_tol = 1e-12
    kmf_max_cycle = 100

    gdf_fname = 'BN_{0:2.2f}_gdf_ints_551.h5'.format(a)
    chkfname = 'BN_{0:2.2f}_551.chk'.format(a)

    ### ************************************************************
    ### DMET settings 
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

    # DMET SCF control
    MaxIter = 6 
    u_tol = 1.0e-4
    E_tol = 5.0e-5
    iter_tol = 4

    # DIIS
    adiis = lib.diis.DIIS()
    adiis.space = 4
    diis_start = 4
    dc = dmet.FDiisContext(adiis.space)

    # solver and mu fit
    nelec_tol = 1.0e-5
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    det = False
    emb_fit_iter = 2000 # embedding fitting
    full_fit_iter = 0
    ytol = 1e-7
    gtol = 1e-3 
    CG_check = False

    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************

    log.section("\nSolving SCF mean-field problem\n")

    gdf = df.GDF(cell, kpts)
    gdf.mesh = np.asarray([7, 7, 63])
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = df.aug_etb(cell, beta=2.3)
    gdf.linear_dep_threshold = 0.
    if not os.path.isfile(gdf_fname):
        gdf.build()

    if os.path.isfile(chkfname):
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:    
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()    
    Eelec_HF = kmf.e_tot -  kmf.energy_nuc()

    log.result("kmf electronic energy: %20.12f", Eelec_HF)

    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    # IAO
    S_ao_ao = kmf.get_ovlp()
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True)

    # HP: I don't know the purpose of this assertion here, I'll check later
    # assert(nval == C_ao_iao_val.shape[-1])

    # idx of orbitals
    aoind = cell.aoslice_by_atom()
    ao_labels = cell.ao_labels()

    labels, B2_labels, virt_labels = get_labels(cell, minao=minao)

    iao_B = get_idx(labels, 0)
    iao_N = get_idx(labels, 1)

    iao_B_val = get_idx(B2_labels, 0)
    iao_N_val = get_idx(B2_labels, 1)

    iao_B_virt = get_idx(virt_labels, 0, offset=len(B2_labels))
    iao_N_virt = get_idx(virt_labels, 1, offset=len(B2_labels))

    iao_B_2s2px2py = [0, 1, 2]
    iao_B_pz = [3]
    iao_N_2s2px2py = [4, 5, 6]
    iao_N_pz = [7]

    # use IAO
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4)

    # Define fragments list:
    def partition(scheme=0):
        if scheme == 0:
            # three fragments B 2s2px2py, N 2s2px2py, and everything
            frags = [[iao_B_2s2px2py,[],0],\
                       [iao_N_2s2px2py,[],0],\
                       [iao_B_pz + iao_N_pz,iao_B_virt + iao_N_virt,0]]

        elif scheme == 1:
            # Unit cell
            frags = [[iao_B_val + iao_N_val, \
                       iao_B_virt + iao_N_virt,0]]

        return frags

    frags = partition(scheme=0)
    Sz_frags = [0, 0, 0]
    rebuild_veff = True
    if len(frags) > 1: rebuild_veff = False

    # vcor initialization
    idx_range = sum([frag_I[0] for frag_I in frags[:2]],[])

    vcor = dmet.VcorLocal_new(restricted, bogoliubov, nscsites, idx_range=idx_range)
    vcor.update(np.zeros(vcor.length()))

    ### ************************************************************
    ### DMET procedure
    ### ************************************************************

    # DMET main loop
    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    if load_frecord:
        dmet.SolveImpHam_with_fitting.load("./frecord")
        
    e_nuc = kmf.energy_nuc()    

    for iteration in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iteration)

        log.section("\nSolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)

        # DMET procedure
        I = 0
        Lat_ls = []
        ImpHam_ls = []
        solver_ls = [] 
        solver_args_ls = []
        imp_idx_ls = []
        basis_ls = []
        H1e_ls = []    
        for frag_I, Sz_I in zip(frags,Sz_frags):
            val, virt, core = frag_I
            Lat = lattice.Lattice(cell, kmesh)
            kpts = Lat.kpts
            nao = Lat.nao
            nkpts = Lat.nkpts
            Lat.set_val_virt_core(val, virt, core)  # this is only used when construct the imp Hamil
            Lat.set_Ham(kmf, gdf, C_ao_lo)

            log.section("\nSolving mean-field problem for fragment %s\n", I)
            rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
            rho = rho.real
            rho_k = Lat.R2k(rho)

            log.section("\nConstructing impurity problem for fragment %s\n", I)
            ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, int_bath=int_bath, max_memory=max_memory)
            ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
            basis_k = Lat.R2k_basis(basis)
            solver_args = {"nelec": min((Lat.ncore+Lat.nval)*2, cell.nelectron*nkpts), \
                           "dm0": dmet.foldRho_k(rho_k, basis_k)}
            if I < 2: 
                # This is the QC solver.
                solver = QCyao(restricted=restricted, tol=1e-6)
                # solver = dmet.impurity_solver.FCI(restricted=restricted, tol=5e-7, max_memory=max_memory, Sz=Sz_I)
                # solver = dmet.impurity_solver.CCSD(restricted=restricted, tol=5e-7, tol_normt=5e-5,max_memory=max_memory, Sz=Sz_I)
            else:
                solver = dmet.impurity_solver.CCSD(restricted=restricted, tol=5e-7, tol_normt=5e-5, max_memory=max_memory, Sz=Sz_I)

            # Store the embedding Hamiltonian for each fragment to pass it to SolveImpHam_with_fitting
            H1e_ls.append(H1e)
            Lat_ls.append(Lat)
            ImpHam_ls.append(ImpHam)
            solver_ls.append(solver)
            solver_args_ls.append(solver_args)
            imp_idx_ls.append(range(len(val + virt)))
            basis_ls.append(basis)
            I += 1

        log.section("\nSolving impurity problems\n",)
        rhoEmb_ls, EnergyEmb_ls, ImpHam_ls, dmu = \
            dmet.SolveImpHam_with_fitting(Lat_ls, Filling, ImpHam_ls, basis_ls, solver_ls, \
            solver_args=solver_args_ls, imp_idx=imp_idx_ls, thrnelec=nelec_tol, \
            delta=delta, step=step)

        # dmet.SolveImpHam_with_fitting.save("./frecord")
        log.section("\nTransform the embedding results to the total system\n",)
        I = 0
        EnergyImp = 0.
        nelecImp = 0.
        vcor_I_list = [] 
        rhoImp_list = []
        for frag_I, Sz_I in zip(frags,Sz_frags):
            rhoImp_I, EnergyImp_I, nelecImp_I = \
                dmet.transformResults(rhoEmb_ls[I], EnergyEmb_ls[I], basis_ls[I], \
                                            ImpHam_ls[I], H1e=H1e_ls[I], lattice=Lat_ls[I], last_dmu=last_dmu, int_bath=int_bath, \
                                            solver=solver_ls[I], solver_args=solver_args_ls[I],
                                            rebuild_veff=rebuild_veff)

            EnergyImp_I *= nscsites
            EnergyImp += EnergyImp_I
            nelecImp += nelecImp_I
            
            if iteration ==0:
                os_energy_ls.append((a,kmf.e_tot, EnergyImp + e_nuc, nelecImp, last_dmu))

            log.section("\nfitting correlation potential for fragment %s\n", I)
            vcor_I = dmet_tool.vcor_to_vcor_I(dmet, restricted, bogoliubov, nscsites, frags, I, vcor)
            if I < 2:
                vcor_I_new, err = dmet.FitVcor(rhoEmb_ls[I], Lat_ls[I], basis_ls[I], \
                        vcor_I, beta, Filling, MaxIter1=emb_fit_iter, MaxIter2=full_fit_iter, method='CG', \
                        imp_fit=imp_fit, det=det, ytol=ytol, gtol=gtol, CG_check=CG_check)    
            else:
                vcor_I_new, err = vcor_I, 0  

            vcor_I_list.append(vcor_I_new)
            rhoImp_list.append(rhoImp_I)
            log.result("Nelec   for fragment %s = %20.12f", I, nelecImp_I)
            log.result("E(DMET) for fragment %s = %20.12f", I, EnergyImp_I)
            I += 1

        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", EnergyImp)

        vcor_new = dmet_tool.vcor_I_to_vcor(dmet, restricted, bogoliubov, nscsites, frags, vcor, vcor_I_list)
        dVcor_per_ele = np.max(np.abs(vcor_new.param - vcor.param))

        dE = EnergyImp - E_old
        E_old = EnergyImp 

        if iteration >= diis_start:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param
            
        

        dVcor_per_ele = np.max(np.abs(pvcor - vcor.param))
        vcor.update(pvcor)
        log.result("Trace of vcor: %20.12f ", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))

        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()

        if dVcor_per_ele < u_tol and abs(dE) < E_tol and iteration > iter_tol :
            conv = True
            break
    if conv:
        log.result("DMET converge.")
    else:
        log.result("DMET does not converge.")

   
    log.result("\nTotal energy\n")
    log.result("HF.  : %20.12f", kmf.e_tot)
    log.result("DMET : %20.12f", EnergyImp + e_nuc)
    sc_energy_ls.append((a,kmf.e_tot, EnergyImp + e_nuc, err, nelecImp, dVcor_per_ele))

