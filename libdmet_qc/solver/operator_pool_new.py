# Copyright (c) Bytedance Inc. 
# SPDX-License-Identifier: GPL-3.0-Only

"""Infrastructure for ADAPT VQE algorithm"""

from libdmet_qc.solver.utils import (spinorb_from_spatial_unrestricted, spinorb_from_spatial_restricted, up_index, down_index, ao2mo_ham)
import pyscf
from pyscf import gto, scf, cc
from openfermion import FermionOperator, jordan_wigner,normal_ordered
from collections import OrderedDict as ordict
import itertools
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner
from openfermion.chem import MolecularData
from openfermion.config import *
from fqe.algorithm.adapt_vqe import OperatorPool, ADAPT
from fqe.algorithm.brillouin_calculator import get_fermion_op
from fqe.fqe_decorators import build_hamiltonian
from fqe.hamiltonians.hamiltonian import Hamiltonian as ABCHamiltonian
from fqe.hamiltonians.restricted_hamiltonian import RestrictedHamiltonian
from fqe.wavefunction import Wavefunction
import fqe
# from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion import (
    make_reduced_hamiltonian,
    InteractionOperator,
)
from fqe.algorithm.brillouin_calculator import (
    get_fermion_op,
    two_rdo_commutator_symm,
    one_rdo_commutator_symm,
)

from fqe.algorithm.algorithm_util import valdemaro_reconstruction

import openfermion as of
import scipy as sp
import numpy as np
from itertools import product
import time
import copy
from typing import List, Tuple, Union, Dict
import os
os.environ['OMP_NUM_THREADS'] = '16'


class OperatorPool_new:

    def __init__(self, norbs: int, occ: List[int], virt: List[int]):
        """
        Routines for defining operator pools
        Args:
            norbs: number of spatial orbitals
            occ: list of indices of the occupied orbitals
            virt: list of indices of the virtual orbitals
        """
        self.norbs = norbs
        self.occ = occ
        self.virt = virt
        self.op_pool: List[of.FermionOperator] = []

    def singlet_t2(self):
        """
        Generate singlet rotations
        T_{ij}^{ab} = T^(v1a, v2b)_{o1a, o2b} + T^(v1b, v2a)_{o1b, o2a} +
                      T^(v1a, v2a)_{o1a, o2a} + T^(v1b, v2b)_{o1b, o2b}
        where v1,v2 are indices of the virtual obritals and o1, o2 are
        indices of the occupied orbitals with respect to the Hartree-Fock
        reference.
        """
        for oo_i in self.occ:
            for oo_j in self.occ:
                for vv_a in self.virt:
                    for vv_b in self.virt:
                        term = of.FermionOperator()
                        for sigma, tau in product(range(2), repeat=2):
                            op = ((2 * vv_a + sigma, 1), (2 * vv_b + tau, 1),
                                  (2 * oo_j + tau, 0), (2 * oo_i + sigma, 0))
                            if (2 * vv_a + sigma == 2 * vv_b + tau or
                                    2 * oo_j + tau == 2 * oo_i + sigma):
                                continue
                            fop = of.FermionOperator(op, coefficient=0.5)
                            fop = fop - of.hermitian_conjugated(fop)
                            fop = of.normal_ordered(fop)
                            term += fop
                        self.op_pool.append(term)

    def generalized_two_body(self):
        """
        Doubles generators each with distinct Sz expectation value.
        """
        for i, j, k, l in product(range(2 * self.norbs), repeat=4):
            if i != j and k != l:
                op = ((i, 1), (j, 1), (k, 0), (l, 0))
                fop_aa = of.FermionOperator(op)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                self.op_pool.append(fop_aa)

    def generalized_two_body_minimal(self):
        """
        Doubles generators each with distinct Sz expectation value.
        """
        for i, j, k, l in product(range(2 * self.norbs), repeat=4):
            if i < j and k < l:
                op = ((i, 1), (j, 1), (k, 0), (l, 0))
                fop_aa = of.FermionOperator(op)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                self.op_pool.append(fop_aa)

    def two_body_sz_adapted(self):
        """
        Doubles generators each with distinct Sz expectation value.
        G^{isigma, jtau, ktau, lsigma) for sigma, tau in 0, 1
        """
        for i, j, k, l in product(range(self.norbs), repeat=4):
            if i < j and k < l:
                op_aa = ((2 * i, 1), (2 * j, 1), (2 * k, 0), (2 * l, 0))
                op_bb = ((2 * i + 1, 1), (2 * j + 1, 1), (2 * k + 1, 0),
                         (2 * l + 1, 0))
                fop_aa = of.FermionOperator(op_aa)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                fop_bb = of.FermionOperator(op_bb)
                fop_bb = fop_bb - of.hermitian_conjugated(fop_bb)
                fop_aa = of.normal_ordered(fop_aa)
                fop_bb = of.normal_ordered(fop_bb)
                self.op_pool.append(fop_aa)
                self.op_pool.append(fop_bb)

            op_ab = ((2 * i, 1), (2 * j + 1, 1), (2 * k + 1, 0), (2 * l, 0))
            fop_ab = of.FermionOperator(op_ab)
            fop_ab = fop_ab - of.hermitian_conjugated(fop_ab)
            fop_ab = of.normal_ordered(fop_ab)
            if not np.isclose(fop_ab.induced_norm(), 0):
                self.op_pool.append(fop_ab)

    def para_unrestricted_uccsd_generator(self, n_qubits, n_electrons_list, th=-1):
        """"Construct the unrestricted ccsd for the unrestricted coupled cluster.
        Also for this one, we utilize this mainly for embedded system. But it can also use for the generally
        unrestricted system, since we set the initial parameters to be very small."""
        params = {}
        if n_qubits % 2 != 0:
            raise ValueError(
                'The total number of spin-orbitals should be even.')
        out = []
        out_tmp = []
        n_spatial_orbitals = int(n_qubits // 2)
        n_occ_alpha, n_occ_beta = int(
            n_electrons_list[0]), int(n_electrons_list[1])
        # n_occupied = int(np.ceil(n_electrons / 2))
        n_virt_alpha = n_spatial_orbitals - n_occ_alpha
        n_virt_beta = n_spatial_orbitals - n_occ_beta

        # Unpack amplitudes, here we assume alpha => spin up => 2*n, while beta => spin down=> 2*n+1
        n_single_amplitudes_alpha = n_occ_alpha * n_virt_alpha
        n_single_amplitudes_beta = n_occ_beta * n_virt_beta

        def generate_spin_unrestricted_single_excitations(n_virt_spin, n_occ_spin, para_offset, spin='alpha'):
            """Generate the unrestricted spin unrestricted single excitations for spin alpha and beta, respectively.
            spin offset, alpha => 0, while beta => 1"""
            out_temp = []
            if spin == 'alpha':
                offset = 0
            else:
                offset = 1

            for i, (p, q) in enumerate(
                    itertools.product(range(n_virt_spin), range(n_occ_spin))):
                # Get indices of spatial orbitals
                virt_spatial = n_occ_spin + p
                occ_spatial = q
                virt_alpha = virt_spatial * 2 + offset
                occ_alpha = occ_spatial * 2 + offset
                single_amps = 1e-6  # assume this is the value
                single_amps_name = 'p' + str(i + para_offset)
                #print("single_amps_name: ", single_amps_name)

                if abs(single_amps) > th:
                    params[single_amps_name] = single_amps
                    fermion_ops1 = FermionOperator(
                        ((occ_alpha, 1), (virt_alpha, 0)), 1)
                    fermion_ops2 = FermionOperator(
                        ((virt_alpha, 1), (occ_alpha, 0)), 1)
                    out_temp.append(
                        [fermion_ops1 - fermion_ops2, single_amps_name])
                    self.op_pool.append(fermion_ops1 - fermion_ops2)
            return out_temp

        # deal with alpha part
        out_temp = generate_spin_unrestricted_single_excitations(
            n_virt_alpha, n_occ_alpha, 0, spin='alpha')
        out.extend(out_temp)
        # deal with beta part
        out_temp = generate_spin_unrestricted_single_excitations(
            n_virt_beta, n_occ_beta, n_single_amplitudes_alpha, spin='beta')
        out.extend(out_temp)

        def generate_same_spin_unrestricted_two_excitations(n_virt_spin, n_occ_spin, para_offset, spin='alpha'):
            """Generate the unrestricted spin unrestricted the operator ap_dagger aq_dagger ar as, 
            here ap_dagger,as belongs to one electron, while aq_dagger and ar belong to the other electron."""
            # Alpha part
            out_temp = []
            if spin == 'alpha':
                offset = 0
            else:
                offset = 1

            for i, ((p, q), (r, s)) in enumerate(
                itertools.product(
                    itertools.combinations(range(n_virt_spin), 2), itertools.combinations(range(n_occ_spin), 2))):

                # Get indices of spatial orbitals
                virt_spatial_1 = n_occ_spin + p
                virt_spatial_2 = n_occ_spin + q
                occ_spatial_2 = r
                occ_spatial_1 = s  # but for the same spin, these two can be change in factor

                virt_1_alpha = virt_spatial_1 * 2 + offset
                occ_1_alpha = occ_spatial_1 * 2 + offset
                virt_2_alpha = virt_spatial_2 * 2 + offset
                occ_2_alpha = occ_spatial_2 * 2 + offset

                double2_amps = 1e-6
                double2_amps_name = 'p' + str(i + para_offset)
                #print("double2_ampts_name is: ", double2_amps_name)

                if abs(double2_amps) > th:
                    params[double2_amps_name] = double2_amps
                    fermion_ops1 = FermionOperator(
                        ((virt_1_alpha, 1), (occ_1_alpha, 0), (virt_2_alpha, 1),
                         (occ_2_alpha, 0)), 1)
                    fermion_ops2 = FermionOperator(
                        ((occ_2_alpha, 1), (virt_2_alpha, 0), (occ_1_alpha, 1),
                         (virt_1_alpha, 0)), 1)
                    out_temp.append(
                        [fermion_ops1 - fermion_ops2, double2_amps_name])
                    self.op_pool.append(fermion_ops1 - fermion_ops2)
            return out_temp

        # two electrons excitation with the same alpha spin
        n_single_amplitudes = n_single_amplitudes_alpha + n_single_amplitudes_beta
        # generate the two excitations within alpha spin
        out_temp = generate_same_spin_unrestricted_two_excitations(
            n_virt_alpha, n_occ_alpha, n_single_amplitudes, spin='alpha')
        out.extend(out_temp)
        n_double2_alpha = int(0.25*n_occ_alpha*(n_occ_alpha-1)
                              * n_virt_alpha*(n_virt_alpha-1))
        out_temp = generate_same_spin_unrestricted_two_excitations(n_virt_beta, n_occ_beta,
                                                                   n_single_amplitudes + n_double2_alpha, spin='beta')
        out.extend(out_temp)

        # Directly hand the two excitations combine with two different spin.
        n_double2_beta = int(0.25*n_occ_beta*(n_occ_beta-1)
                             * n_virt_beta*(n_virt_beta-1))
    #     print("n_double2_alpha2: ", n_double2_beta)

        # generate the operator ap_dagger aq_dagger ar as, ap_dagger,as belongs to one alpha, while aq_dagger, ar
        # belong to the second electrons.
        for i, ((p, s), (q, r)) in enumerate(
                itertools.product(
                    itertools.product(range(n_virt_alpha), range(n_occ_alpha)), itertools.product(range(n_virt_beta), range(n_occ_beta)))):

            # Get indices of spatial orbitals
            virt_spatial_1 = n_occ_alpha + p
            virt_spatial_2 = n_occ_beta + q
            occ_spatial_2 = r
            occ_spatial_1 = s

            virt_1_alpha = virt_spatial_1 * 2
            occ_1_alpha = occ_spatial_1 * 2
            virt_2_beta = virt_spatial_2 * 2 + 1
            occ_2_beta = occ_spatial_2 * 2 + 1

            double2_amps = 1e-5
            double2_amps_name = 'p' + \
                str(i + n_single_amplitudes + n_double2_alpha + n_double2_beta)

            if abs(double2_amps) > th:
                params[double2_amps_name] = double2_amps
                fermion_ops1 = FermionOperator(
                    ((virt_1_alpha, 1), (occ_1_alpha, 0), (virt_2_beta, 1),
                     (occ_2_beta, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occ_2_beta, 1), (virt_2_beta, 0), (occ_1_alpha, 1),
                     (virt_1_alpha, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, double2_amps_name])
                self.op_pool.append(fermion_ops1 - fermion_ops2)

        n_double2_alpha_beta = int(
            n_occ_beta*(n_occ_beta-1)*n_virt_beta*(n_virt_beta-1))

        return out, params


    def para_uccsd_singlet_generator(self, n_qubits, n_electrons, th=-1):
        #n_qubits = mol.n_qubits
        #n_electrons = mol.n_electrons
        params = {}
        if n_qubits % 2 != 0:
            raise ValueError(
                'The total number of spin-orbitals should be even.')
        out = []
        out_tmp = []
        n_spatial_orbitals = n_qubits // 2
        n_occupied = int(np.ceil(n_electrons / 2))
        n_virtual = n_spatial_orbitals - n_occupied

        # Unpack amplitudes
        n_single_amplitudes = n_occupied * n_virtual
        # Generate excitations
        spin_index_functions = [up_index, down_index]
        # Generate all spin-conserving single and double excitations derived
        # from one spatial occupied-virtual pair
        for i, (p, q) in enumerate(
                itertools.product(range(n_virtual), range(n_occupied))):

            # Get indices of spatial orbitals
            virtual_spatial = n_occupied + p
            occupied_spatial = q
            virtual_up = virtual_spatial * 2
            occupied_up = occupied_spatial * 2
            virtual_down = virtual_spatial * 2 + 1
            occupied_down = occupied_spatial * 2 + 1

            single_amps = 1e-5
            double1_amps = 1e-5
            single_amps_name = 'p' + str(i)
            double1_amps_name = 'p' + str(i + n_single_amplitudes)

            if abs(single_amps) > th:
                #             cnt +=1
                # deal with spin up part
                params[single_amps_name] = single_amps
                fermion_ops1 = FermionOperator(
                    ((occupied_up, 1), (virtual_up, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((virtual_up, 1), (occupied_up, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, single_amps_name])

                # deal with spin down part
                fermion_ops1 = FermionOperator(
                    ((occupied_down, 1), (virtual_down, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((virtual_down, 1), (occupied_down, 0)), 1)

                out.append([fermion_ops1 - fermion_ops2, single_amps_name])

            # Generate double excitation
            if abs(double1_amps) > th:
                params[double1_amps_name] = double1_amps
                fermion_ops1 = FermionOperator(
                    ((virtual_up, 1), (occupied_up, 0), (virtual_down, 1),
                     (occupied_down, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_down, 1), (virtual_down, 0),
                     (occupied_up, 1), (virtual_up, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, double1_amps_name])
                self.op_pool.append(fermion_ops1 - fermion_ops2)
    #         print("cnt is: ",cnt)
        out.extend(out_tmp)
        out_tmp = []
        # Generate all spin-conserving double excitations derived
        # from two spatial occupied-virtual pairs
        for i, ((p, q), (r, s)) in enumerate(
                itertools.combinations(
                    itertools.product(range(n_virtual), range(n_occupied)), 2)):

            # Get indices of spatial orbitals
            virtual_spatial_1 = n_occupied + p
            occupied_spatial_1 = q
            virtual_spatial_2 = n_occupied + r
            occupied_spatial_2 = s

            virtual_1_up = virtual_spatial_1 * 2
            occupied_1_up = occupied_spatial_1 * 2
            virtual_2_up = virtual_spatial_2 * 2 + 1
            occupied_2_up = occupied_spatial_2 * 2 + 1

            double2_amps = 1e-5
            double2_amps_name = 'p' + str(i + 2 * n_single_amplitudes)

            # Generate double excitations
            for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
                # Get the functions which map a spatial orbital index to a
                # spin orbital index
                index_a = spin_index_functions[spin_a]
                index_b = spin_index_functions[spin_b]

                # Get indices of spin orbitals
                virtual_1_a = index_a(virtual_spatial_1)
                occupied_1_a = index_a(occupied_spatial_1)
                virtual_2_b = index_b(virtual_spatial_2)
                occupied_2_b = index_b(occupied_spatial_2)
                if virtual_1_a == virtual_2_b or occupied_1_a == occupied_2_b:
                    pass
                else:
                    if abs(double2_amps) > th:
                        params[double2_amps_name] = double2_amps
                        fermion_ops1 = FermionOperator(
                            ((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1),
                             (occupied_2_b, 0)), 1)
                        fermion_ops2 = FermionOperator(
                            ((occupied_2_b, 1), (virtual_2_b, 0), (occupied_1_a, 1),
                             (virtual_1_a, 0)), 1)
                        out.append(
                            [fermion_ops1 - fermion_ops2, double2_amps_name])
                        self.op_pool.append(fermion_ops1 - fermion_ops2)
        return out, params

    def one_body_sz_adapted(self):
        # alpha-alpha rotation
        # beta-beta rotation
        for i, j in product(range(self.norbs), repeat=2):
            if i > j:
                op_aa = ((2 * i, 1), (2 * j, 0))
                op_bb = ((2 * i + 1, 1), (2 * j + 1, 0))
                fop_aa = of.FermionOperator(op_aa)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                fop_bb = of.FermionOperator(op_bb)
                fop_bb = fop_bb - of.hermitian_conjugated(fop_bb)
                fop_aa = of.normal_ordered(fop_aa)
                fop_bb = of.normal_ordered(fop_bb)
                self.op_pool.append(fop_aa)
                self.op_pool.append(fop_bb)


class ADAPT_new:

    def __init__(self,
                 oei: np.ndarray,
                 tei: np.ndarray,
                 operator_pool,
                 n_alpha: int,
                 n_beta: int,
                 iter_max=50,
                 verbose=True,
                 restricted=True,
                 stopping_epsilon=1.0E-3,
                 delta_e_eps=1.0E-6):
        """
        ADAPT-VQE object.
        Args:
            oei: one electron integrals in the spatial basis
            tei: two-electron integrals in the spatial basis
            operator_pool: Object with .op_pool that is a list of antihermitian
                           FermionOperators
            n_alpha: Number of alpha-electrons
            n_beta: Number of beta-electrons
            iter_max: Maximum ADAPT-VQE steps to take
            verbose: Print the iteration information
            stopping_epsilon: define the <[G, H]> value that triggers stopping
        """
#         elec_hamil = RestrictedHamiltonian((oei, np.einsum("ijlk", -0.5 * tei)))
        if restricted:
            soei, stei = spinorb_from_spatial_restricted(oei, tei)
        else:
            soei, stei = spinorb_from_spatial_unrestricted(oei, tei)
        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
        molecular_hamiltonian = InteractionOperator(0, soei, 0.25 * astei)

        reduced_ham = make_reduced_hamiltonian(molecular_hamiltonian,
                                               n_alpha + n_beta)
        self.reduced_ham = reduced_ham
        dim = oei.shape[-1]
        self.k2_ham = of.get_fermion_operator(reduced_ham)
        self.k2_fop = build_hamiltonian(self.k2_ham,
                                        dim,
                                        conserve_number=True)
        # self.elec_hamil = elec_hamil # seems only this one provide the dim
        self.iter_max = iter_max
        self.sdim = dim
        # change to use multiplicity to derive this for open shell
        self.nalpha = n_alpha
        self.nbeta = n_beta
        self.sz = self.nalpha - self.nbeta
        self.nele = self.nalpha + self.nbeta
        self.verbose = verbose
        self.operator_pool = operator_pool
        self.stopping_eps = stopping_epsilon
        self.delta_e_eps = delta_e_eps
        norbs = oei.shape[1]
        fqe_wf_initial = fqe.Wavefunction([[self.nele, self.sz, norbs]])
        fqe_ordm, fqe_trdm = fqe_wf_initial.sector(
            (self.nele, self.sz)).get_openfermion_rdms()
        self.one_rdm = fqe_ordm
        self.two_rdm = fqe_trdm
        self.wf = None
        
    
    def adapt_vqe(self,
                  initial_wf: Wavefunction,
                  opt_method: str = 'L-BFGS-B',
                  opt_options = None,
                  num_opt_var = None,
                  v_reconstruct: bool = True,
                  num_ops_add: int = 1):
        """
        Run ADAPT-VQE using
        Args:
            initial_wf: Initial wavefunction at the start of the calculation
            opt_method: scipy optimizer to use
            opt_options: options  for scipy optimizer
            v_reconstruct: use valdemoro reconstruction
            num_ops_add: add this many operators from the pool to the
                         wavefunction
        """
        if opt_options is None:
            opt_options = {}
        self.num_opt_var = num_opt_var
        operator_pool = []
        operator_pool_fqe: List[ABCHamiltonian] = []
        existing_parameters: List[float] = []
        self.gradients = []
        self.energies = [initial_wf.expectationValue(self.k2_fop)]
        iteration = 0
        print("*******Call Adapt VQE.******")
        while iteration < self.iter_max:
            # get current wavefunction
            wf = copy.deepcopy(initial_wf)
            for fqe_op, coeff in zip(operator_pool_fqe, existing_parameters):
                #print("fqe_op: ", fqe_op)
                #print("fqe_coeff: ", coeff)
                wf = wf.time_evolve(coeff, fqe_op)

            # calculate rdms for grad
            _, tpdm = wf.sector((self.nele, self.sz)).get_openfermion_rdms()
            if v_reconstruct:
                d3 = 6 * valdemaro_reconstruction(tpdm / 2, self.nele)
            else:
                d3 = wf.sector((self.nele, self.sz)).get_three_pdm()

            # get ACSE Residual and 2-RDM gradient
            acse_residual = two_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm, d3)
            one_body_residual = one_rdo_commutator_symm(
                self.reduced_ham.two_body_tensor, tpdm)

            # calculate grad of each operator in the pool
            pool_grad = []
            for operator in self.operator_pool.op_pool:
                grad_val = 0
                operator = normal_ordered(operator)#Default this is normal order case.
                for op_term, coeff in operator.terms.items():
                    idx = [xx[0] for xx in op_term]
                    if len(idx) == 4:
                        grad_val += acse_residual[tuple(idx)] * coeff
                    elif len(idx) == 2:
                        grad_val += one_body_residual[tuple(idx)] * coeff
                pool_grad.append(grad_val)
            #print("The gradient of operators are", pool_grad)
            max_grad_terms_idx = \
                np.argsort(np.abs(pool_grad))[::-1][:num_ops_add]
            print("Maximum gradient:", max(np.abs(pool_grad)))

            pool_terms = [
                self.operator_pool.op_pool[i] for i in max_grad_terms_idx
            ]
            operator_pool.extend(pool_terms)
            print("Operator pool: ", operator_pool)
            fqe_ops: List[ABCHamiltonian] = []
            for f_op in pool_terms:
                fqe_ops.append(
                    build_hamiltonian(1j * f_op,
                                      self.sdim,
                                      conserve_number=True))
            operator_pool_fqe.extend(fqe_ops)
            existing_parameters.extend([0] * len(fqe_ops))

            if self.num_opt_var is not None:
                if len(operator_pool_fqe) < self.num_opt_var:
                    pool_to_op = operator_pool_fqe
                    params_to_op = existing_parameters
                    current_wf = copy.deepcopy(initial_wf)
                else:
                    pool_to_op = operator_pool_fqe[-self.num_opt_var:]
                    params_to_op = existing_parameters[-self.num_opt_var:]
                    current_wf = copy.deepcopy(initial_wf)
                    for fqe_op, coeff in zip(
                            operator_pool_fqe[:-self.num_opt_var],
                            existing_parameters[:-self.num_opt_var]):
                        current_wf = current_wf.time_evolve(coeff, fqe_op)

                new_parameters, current_e = self.optimize_param(
                    pool_to_op,
                    params_to_op,
                    current_wf,
                    opt_method,
                    opt_options=opt_options)

                if len(operator_pool_fqe) < self.num_opt_var:
                    existing_parameters = new_parameters.tolist()
                else:
                    existing_parameters[-self.num_opt_var:] = \
                        new_parameters.tolist()
            else:
                new_parameters, current_e = self.optimize_param(
                    operator_pool_fqe,
                    existing_parameters,
                    initial_wf,
                    opt_method,
                    opt_options=opt_options)
                existing_parameters = new_parameters.tolist()
                
            

            if self.verbose:
                print("{: 5d}\t{: 5.15f}\t{: 5.15f}".format(
                    iteration, current_e, max(np.abs(pool_grad))))
            self.energies.append(current_e)
            self.gradients.append(pool_grad)
#             if max(np.abs(pool_grad)) < self.stopping_eps or np.abs(
#                      self.energies[-2] - self.energies[-1]) < self.delta_e_eps:
            if  np.abs(self.energies[-2] - self.energies[-1]) < self.delta_e_eps:
#             if max(np.abs(pool_grad)) < self.stopping_eps:
                print("We enter here.", self.energies)
                break
            iteration += 1
            
        return new_parameters, self.energies[-1]

    def uccsd_vqe(self,
                  initial_wf: Wavefunction,
                  operator_pool: List[ABCHamiltonian],
                  initial_parameters: List[float],
                  opt_method: str = 'L-BFGS-B',
                  opt_options=None):
        """
        Run ADAPT-VQE using
        Args:
            initial_wf: Initial wavefunction at the start of the calculation
            opt_method: scipy optimizer to use
            opt_options: options  for scipy optimizer
            v_reconstruct: use valdemoro reconstruction
            num_ops_add: add this many operators from the pool to the
                         wavefunction
        """
        if opt_options is None:
            opt_options = {}
        # self.num_opt_var = num_opt_var
        # operator_pool = []
        operator_pool_fqe = operator_pool
        existing_parameters = initial_parameters
        # self.gradients = []
        # self.energies = [initial_wf.expectationValue(self.k2_fop)]
        #         for fqe_op, coeff in zip(operator_pool_fqe, existing_parameters):
        #             wf = wf.time_evolve(coeff, fqe_op)
        opt_para, opt_energy = self.optimize_param(
            operator_pool_fqe,
            existing_parameters,
            initial_wf,
            opt_method,
            opt_options)
        return opt_para, opt_energy

    def optimize_param(
            self,
            pool: Union[List[of.FermionOperator], List[ABCHamiltonian]],
            existing_params: Union[List, np.ndarray],
            initial_wf: Wavefunction,
            opt_method: str,
            opt_options=None) -> Tuple[np.ndarray, float]:
        """Optimize a wavefunction given a list of generators
        Args:
            pool: generators of rotation
            existing_params: parameters for the generators
            initial_wf: initial wavefunction
            opt_method: Scpy.optimize method
        """
        if opt_options is None:
            opt_options = {}

        def cost_func(params):
            assert len(params) == len(pool)
            # compute wf for function call
            wf = copy.deepcopy(initial_wf)
            time1 = time.time()
            for op, coeff in zip(pool, params):
                # print("Now the op is: ",op)
                if np.isclose(coeff, 0):
                    continue
                if isinstance(op, ABCHamiltonian):
                    #print("Use this fqe_op!")
                    # print("op is: ",op)
                    fqe_op = op
                else:
                   # print("Found a OF Hamiltonian")
                    fqe_op = build_hamiltonian(1j * op,
                                               self.sdim,
                                               conserve_number=True)
                if isinstance(fqe_op, ABCHamiltonian):
                    # print("Begin the time evolution")
                    # print("coeff: ",coeff)
                    # print("fqe_op: ",fqe_op.terms()) # this is a [] empty terms
                    try:
                        wf = wf.time_evolve(coeff, fqe_op)
                    except:
                        continue
                    # print("End the time evolution")
                else:
                    raise ValueError("Can't evolve operator type {}".format(
                        type(fqe_op)))

            # compute the energy for this system
            exp = wf.expectationValue(self.k2_fop).real

            # refactor this part of codes in compute gradients
            time2 = time.time()
            # print("State construction times: ", time2-time1)
            grad_vec = np.zeros(len(params), dtype=np.complex128)
            # avoid extra gradient computation if we can
            # we can do this in a reverse way.
            pool_reverse = pool[::-1]
            new_params = params[::-1]
            if opt_method not in ['Nelder-Mead', 'COBYLA']:
                begin = time.time()
                # grad_wf = copy.deepcopy(wf)
                #                     brawf_new = grad_wf.apply(self.k2_fop)
                # for pidx, _ in enumerate(params):
                single_grad_begin = time.time()
                # evolve e^{iG_{n-1}g_{n-1}}e^{iG_{n-2}g_{n-2}}x
                # G_{n-3}e^{-G_{n-3}g_{n-3}...|0>

                # In this case, we only need one for loop.

                grad_wf = copy.deepcopy(wf)
                brawf_new = wf.apply(self.k2_fop)
                for gidx, (op, coeff) in enumerate(zip(pool_reverse, new_params)):
                    # print("Now the gidx is: ", gidx)
                    if isinstance(op, ABCHamiltonian):
                        fqe_op = op
                    else:
                        # print("here enter the parameters: ")
                        #print("Construct the hamiltonians: ")
                        cons_time = time.time()
                        fqe_op = build_hamiltonian(1j * op,
                                                   self.sdim,
                                                   conserve_number=True)
                        cons_time2 = time.time()
                        #print("Construction hamiltonian takes: ", cons_time2 - cons_time2)
                    if not np.isclose(coeff, 0):

                        grad_time1 = time.time()
                        try:
                            # back propgation in this case.
                            #print("Evalutate gradients here.")
                            grad_wf = grad_wf.time_evolve(coeff, fqe_op)
                            brawf_new = brawf_new.time_evolve(coeff, fqe_op)
                        except:
                            continue
                        # if looking at the pth parameter then apply the
                        # operator to the state
                        grad_time2 = time.time()
                        #print("Grad time2:", grad_time2 - grad_time1)

                    # grad_val = grad_wf.expectationValue(self.elec_hamil,
                    # brawfn=wf)
                    # print("Out of the for loops!")
                    time5 = time.time()
                    grad_val = grad_wf.expectationValue(
                        fqe_op, brawfn=brawf_new)
                    #print("grad value is: ", grad_val)
                    #print("****gidx is****: ", gidx)
                    grad_vec[gidx] = -1j * grad_val + 1j * grad_val.conj()
                    time6 = time.time()

                end = time.time()
                #print("end-begin:", end - begin)
                #print("gradients: ", np.array(grad_vec.real[::-1], order='F'))

            print("The current energy is", wf.expectationValue(self.k2_fop).real)
            self.wf = wf
            return (wf.expectationValue(self.k2_fop).real,
                    np.array(grad_vec.real[::-1], order='F'))

        res = sp.optimize.minimize(cost_func,
                                   existing_params,
                                   method=opt_method,
                                   jac=True,
                                   options=opt_options)
        return res.x, res.fun


#************************** Some simple test in this example **************************
#******************************************************
def build_lih_moleculardata(bond_len):
    """Generate the data for the tests."""
    #bond_len = 0.2

    atom1 = 'Li'
    atom2 = 'H'
    coord1 = (0, 0, 0)
    coord2 = (0, 0, bond_len)
    #coordinate = [(0.0, 0.0, 0.0 + i * bond_len) for i in range(10)]
    geometry = [(atom1, coord1), (atom2, coord2)]
    print(geometry)
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    #     bond_len = 1.45
    molecule = MolecularData(
        geometry, basis, multiplicity, charge, description=str(bond_len))

    molecule = run_pyscf(molecule, run_scf=1, run_ccsd=1, run_fci=1)
    # can calculate on arbitrary bond length if package openfermionpyscf is available,
    # only existing molecule data can be used
    molecule.load()

    return molecule


def build_h6_moleculardata(bond_len):
    """Generate the data for the tests."""
    #bond_len = 0.2

    atom = ['H'] * 6
    coordinate = [(0.0, 0.0, 0.0 + i * bond_len) for i in range(6)]
    geometry = [(atom[i], coordinate[i]) for i in range(6)]
    print(geometry)
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    #     bond_len = 1.45
    molecule = MolecularData(
        geometry, basis, multiplicity, charge, description=str(bond_len))

    molecule = run_pyscf(molecule, run_scf=1, run_ccsd=1, run_fci=1)
    # can calculate on arbitrary bond length if package openfermionpyscf is available,
    # only existing molecule data can be used
    molecule.load()

    return molecule


def NaH(bond_len):
    atom_1 = 'Li'
    atom_2 = 'H'
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (0.0, 0.0, bond_len)
    NaH = [(atom_1, coordinate_1), (atom_2, coordinate_2)]

    molecule = MolecularData(NaH, basis, multiplicity,
                             charge, description=str(bond_len))
    molecule = run_pyscf(molecule, run_scf=1, run_ccsd=1, run_fci=1)
    # can calculate on arbitrary bond length if package openfermionpyscf is available,
    # only existing molecule data can be used
    molecule.load()
    return molecule


def test_restricted_uccsd(bond_len):
    molecule = NaH(bond_len)
    print("the fci energy is", molecule.fci_energy)
    n_electrons = molecule.n_electrons
    oei, tei = molecule.get_integrals()

    # print("double electron integral is", tei)
    # print(np.allclose(ele[0, :, :, :], tei[:, 0, :, :]))
    #print(np.allclose(ele, tei))
    norbs = molecule.n_orbitals
    print("the number of norbs is", norbs)
    nalpha = molecule.n_electrons // 2
    nbeta = nalpha
    sz = nalpha - nbeta
    occ = list(range(nalpha))
    virt = list(range(nalpha, norbs))
    #     print("occ: ", occ)
    #     print("virt: ", virt)
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.chem.make_reduced_hamiltonian(molecular_hamiltonian,
                                                   molecule.n_electrons)
    begin = time.time()
    fqe_wf = fqe.Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    fqe_wf.set_wfn(strategy='hartree-fock')

    sop = OperatorPool_new(norbs, occ, virt)
    print("the number of orbital is", norbs)
    print("the number of occ is", occ)
    print("the number of virt is", virt)
    # sop.operators_from_dingshun()
    sop.para_uccsd_singlet_generator(norbs*2, n_electrons, -1)
    print(len(sop.op_pool))
    adapt = ADAPT_new(oei, tei, sop, nalpha, nbeta, verbose=False)
    adapt.verbose = True
    existing_parameter = []
    existing_parameter.extend([0] * len(sop.op_pool))
    opt_para, opt_energy = adapt.uccsd_vqe(
        initial_wf=fqe_wf, operator_pool=sop.op_pool, initial_parameters=existing_parameter)
    """
    adapt.adapt_vqe(fqe_wf)
    one_rdm = adapt.get_rdm1()
    two_rdm = adapt.get_rdm2()
    
    total_energy_rdm = np.einsum('ij,ij->', (soei) , one_rdm) + 0.50 * np.einsum('ijkl,ijkl->', stei, two_rdm)
    print("the total_energy_rdm is", total_energy_rdm)
    total_energy_rdm = molecule.nuclear_repulsion + total_energy_rdm
    """
    end = time.time()
    #("Now the time is: ", end - begin)
    print("The nuclear_repulsion is:", molecule.nuclear_repulsion)
    #print(bond_len, molecule.fci_energy, adapt.energies[-1] + molecule.nuclear_repulsion, total_energy_rdm, end - begin)
    # return (bond_len, molecule.fci_energy, adapt.energies[-1] + molecule.nuclear_repulsion, total_energy_rdm, end - begin)
    print(bond_len, molecule.fci_energy, opt_energy +
          molecule.nuclear_repulsion, end - begin)
    spin_rdm = adapt.wf.sector((adapt.nele, adapt.sz)).get_spin_opdm()
    print("****Spin rdm****: ", spin_rdm)

    return (bond_len, molecule.fci_energy, opt_energy + molecule.nuclear_repulsion, end - begin)

def test_unrestricted_ccsd(bond_length):
    """Treat H4 as an example."""
    mol = pyscf.M(
                atom = 'H 0 0 0; H 0 0 {}; H 0 0 {}; H 0 0 {}'.format(bond_length,  2*bond_length, 3*bond_length),
                basis = 'sto-3g',
                spin = 2
            )
    #         mol = pyscf.M(
    #             atom = 'H 0 0 0; H 0 0 {}'.format(item),
    #             basis = '321g',
    #             spin = 2
    #         )
        #     mol.atom = [['O',[0.0,0.0,0.0]],['H',[0.0,0.0,0.99]]]
        #         mol.spin = 1

    mf = mol.UHF().run()
    mycc = mf.CCSD().run()
    print('CCSD correlation energy', mycc.e_corr)
    print("CCSD energy is: ", mycc.e_tot)


    # Do the quantum parts
    oei, tei = ao2mo_ham(mf, compact=False)
    norbs = mol.nao
    n_electrons = sum(mol.nelec)
    n_qubits = 2*norbs
    nalpha, nbeta = mol.nelec
    sz = nalpha - nbeta
    # the above two are fake parameters for now
    occ = list(range(nalpha))
    virt = list(range(nalpha, norbs))
    #     print("occ: ", occ)
    #     print("virt: ", virt)
    soei, stei = spinorb_from_spatial_unrestricted(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.chem.make_reduced_hamiltonian(molecular_hamiltonian,
                                                n_electrons)
    begin = time.time()
    fqe_wf = fqe.Wavefunction([[n_electrons, sz, norbs]])
    fqe_wf.set_wfn(strategy = 'hartree-fock')
    print("fqe_wf: ", fqe_wf.print_wfn())
    sop = OperatorPool_new(norbs, occ, virt)
    # print("the number of orbital is", norbs)
    # print("the number of occ is", occ)
    # print("the number of virt is", virt)
    # sop.operators_from_dingshun()
    sop.para_unrestricted_uccsd_generator(n_qubits, [nalpha, nbeta], -1)
    # sop.para_uccsd_singlet_generator(n_qubits, nalpha,-1)
    # sop.two_body_sz_adapted()
    print(len(sop.op_pool))
    restricted=False
    adapt = ADAPT_new(oei, tei, sop, nalpha, nbeta, restricted=restricted, verbose=False)
    adapt.verbose = True
    existing_parameter = []
    existing_parameter.extend([0] * len(sop.op_pool))
    opt_para, opt_energy = adapt.uccsd_vqe(initial_wf = fqe_wf, operator_pool = sop.op_pool, initial_parameters = existing_parameter)
    end = time.time()
    print(bond_length, mycc.e_tot, opt_energy + mol.energy_nuc(), end - begin)
        # return adapt.energies[-1]
    spin_rdm = adapt.wf.sector((adapt.nele, adapt.sz)).get_spin_opdm()
    print("****Spin rdm****: ", spin_rdm)
    return opt_energy + mol.energy_nuc()


if __name__ == "__main__":
    """
    e_list = []
    for bd in [i * 0.1 + 0.5 for i in range(5, 10, 1)]:
        try:
            energy = test_adapt_vqe(bd, 200)
            #energy = build_h4_moleculardata(bd).fci_energy
        except KeyError:
            print('vqe is not converged!!!!')

        e_list.append(energy)
        print('e_list:', e_list)
    """
    bond_len = 1.6
    energy = test_restricted_uccsd(bond_len)
    print("the energy by uccsd is", energy)

    energy = test_unrestricted_ccsd(bond_len)
    print("the energy by unrestricted uccsd is", energy)

