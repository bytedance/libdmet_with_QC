# Copyright (c) Bytedance Inc. 
# SPDX-License-Identifier: GPL-3.0-Only

"""
QC impurity solver.
"""

import numpy as np
import scipy.linalg as la
from libdmet.utils import logger as log
# from pyscf.fci import direct_spin1, direct_uhf, cistring
import pyscf.lib.logger as pyscflogger
from libdmet.solver import scf
from libdmet.solver.scf import ao2mo_Ham, restore_Ham
from libdmet.basis_transform.make_basis import transform_rdm1_to_ao_mol, transform_rdm2_to_ao_mol
from libdmet.utils.misc import mdot

import openfermion
from openfermion import InteractionOperator,jordan_wigner, FermionOperator
# Initialize the quantum solver with ProjectQ For simple test.
import projectq
from projectq import MainEngine
from projectq.ops import X, All, Measure, QubitOperator

from scipy.linalg import eigh
import time

# Example for hwo to use this python call julia https://pyjulia.readthedocs.io/en/latest/usage.html
import julia
from julia.api import Julia
# jl =julia.Julia(compiled_modules=False)
jl = Julia(compiled_modules=False)
from julia import Main
solver_path = "/opt/tiger/pdmet_test/softwares/miniconda3/envs/libdmet_qc/lib/python3.8/site-packages/libdmet_qc/solver/CPU_QCsolver_UCCSD.jl"
Main.include(solver_path)
print("***Import periodical_solver.jl Successfully***")


op_dict = {"X": np.array([[0, 1], [1, 0]]),
           "Y": np.array([[0, -1j], [1j, 0]]),
           "Z": np.array([[1, 0], [0, -1]]),
           "Id": np.eye(2)
           }


def parse_op_index(op_index_list):
    """parse the op_index_list into dictionary with proper op and its corresponding index."""
    op_index_dict = {}
    for (idx, op) in op_index_list:
        op_index_dict[idx] = op
    return op_index_dict


# Now write a function to directly diagolize it, so this maybe a direct comparations.
# So first, we need to tranform it into matrix form
# First write the single function for only one terms in this particular case
def operator_to_matrix(n_qubits, op):
    """Transform the operator from qubit operator to matrix for further directly diagonazition,
    n_qubits is the system largest qubits. This should be always larger than that of the first terms."""
    if not isinstance(op, (QubitOperator, openfermion.ops.QubitOperator)):
        raise TypeError("Unsupported type of error")

    all_keys = list(op.terms.keys())
    coeff = list(op.terms.values())
    #     print("all keys: ", all_keys)
    mat = np.zeros((1 << n_qubits, 1 << n_qubits), dtype=np.complex128)
    for i in range(0, len(all_keys)):
        if all_keys[i] == ():  # This is a constant terms
            mat += np.eye((1 << n_qubits)) * coeff[i]
        else:
            item = all_keys[i]
            op_idx_dict = parse_op_index(item)
            #             print("op_idx_dict: ", op_idx_dict)
            # parse this one into dict and make it idenity where no index shows
            mat_temp = 1
            for idx in range(n_qubits):
                #                 print("idx is: ",idx)
                op = op_idx_dict.get(idx, "Id")
                op_mat = op_dict[op]
                mat_temp = np.kron(mat_temp, op_mat)
            mat += mat_temp * coeff[i]
    return mat


class QCyao(object):
    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, \
                 restricted=False, Sz=0, tol=1e-10, max_cycle=200, \
                 max_memory=40000, compact_rdm2=False, scf_newton=True):
        """
        Quantum Computing solver support for the Couple cluster related quantum ansatz.
        """
        self.restricted = restricted
        self.Sz = Sz
        self.conv_tol = tol
        self.max_memory = max_memory
        self.max_cycle = max_cycle
        self.conv_tol = self.conv_tol

        # First performs this unrestricted calculations in this case.
        self.scfsolver = scf.SCF(newton_ah=scf_newton)

        # Onepdm adn twopdm is generally what we called the 1RDM and 2RDM in this case.
        self.onepdm = None
        self.twopdm = None
        self.compact_rdm2 = compact_rdm2  # consider symm of rdm2 # if we have.
        self.optimized = False

        # UHF related to construct the initial unrestricted Hartree-Fock state in this case
        self.n_qubits = None
        self.alpha_elec = None
        self.n_electrons = None
        self.n_orbitals = None
        self.beta_elec = None
        # For the embedded solver.
        self.fermion_hamil = None
        self.qubit_hamil = None
        self.julia_hamil = None # change the qubit_hamil into the julia interface

        # store the optimal opt_circuit for 1RDM and 2RDMs
        self.opt_circuit = None
        self.opt_param = None
        self.para_key_list = None # should be a list

    def run(self, Ham, nelec=None, guess=None, calc_rdm2=False, \
            pspace_size=800, Mu=None, **kwargs):
        """
        Main function of the solver.
        """
        log.info("Quantum Computing solver Run")
        spin = Ham.H1["cd"].shape[0]
        if spin > 1:
            assert not self.restricted
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            else:
                raise ValueError
        nelec_a = (nelec + self.Sz) // 2  # currently treat the restricted for the comparasions for now.
        nelec_b = (nelec - self.Sz) // 2
        assert (nelec_a >= 0) and (nelec_b >= 0) and (nelec_a + nelec_b == nelec)
        self.nelec = (nelec_a, nelec_b)

        # first do a mean-field calculation
        log.debug(1, "Quantum Computing solver: mean-field")
        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)

        # Perform the scf for the corresponding restricted and unrestricted hamiltonian for this system.
        self.scfsolver.set_system(nelec, self.Sz, False, self.restricted, \
                                  max_memory=self.max_memory)
        self.scfsolver.set_integral(Ham)
        E_HF, rhoHF = self.scfsolver.HF(tol=min(1e-9, self.conv_tol * 0.1), \
                                        MaxIter=scf_max_cycle, InitGuess=dm0, Mu=Mu)

        log.debug(1, "Quantum Computing solver: mean-field converged: %s", self.scfsolver.mf.converged)
        # What does this following codes for in this case.
        log.debug(2, "Quantum Computing solver: mean-field rdm1: \n%s", self.scfsolver.mf.make_rdm1())

        print("Check out the mf.mo_coeff for this particularly setup: ", self.scfsolver.mf.mo_coeff.shape)
        # Actually, inorder to make the two part different, we should provide Sz not equal to zero and consider
        # Different initial state for the alpha and beta part for this problem.
#         assert np.allclose(self.scfsolver.mf.mo_coeff[0], self.scfsolver.mf.mo_coeff[1])
        print("Check out the mf.mo_coeff for this particularly setup: ", self.scfsolver.mf.mo_coeff)
        print("Check out the mf.mo_occ for this particularly setup: ", self.scfsolver.mf.mo_occ)
        if self.restricted:  # RHF-FCI
            print("Coeff is: ",self.scfsolver.mf.mo_occ)
            self.n_electrons = int(sum(self.scfsolver.mf.mo_occ))
        else:
            print("Check out the mf.mo_coeff for this particularly setup: ", self.scfsolver.mf.mo_coeff)
            self.alpha_elec, self.beta_elec = int(sum(self.scfsolver.mf.mo_occ[0])), int(sum(self.scfsolver.mf.mo_occ[1]))
            self.n_electrons = [self.alpha_elec, self.beta_elec]
            print("Now the alpha part is: ", self.alpha_elec)
            print("Now the beta part is: ", self.beta_elec)
        # Check the whether alpha part is equal to the beta part
        # self.n_electrons = int(sum(self.scfsolver.mf.mo_occ))
        # This step basically transform it into the new basis for e
        Ham = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff)
        print("Call this Quantum computing solver and perform this transformations.")

        if Ham.restricted:  # RHF-FCI
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
            print("H1 shape: ", h1.shape)
            print("H2 shape: ", h2.shape)
            #Need to check if it is compact for this case
            norb = h1.shape[0]
            if len(h2.shape)!=4:
                from pyscf import ao2mo
                h2 = ao2mo.restore(1, h2, norb)
            print("Now H2 shape: ", h2.shape)
            # Call the function to generate the restricted hamiltonian for this part.
            self.fermion_hamil = self.construct_restricted_hamiltonian(h1, h2)
        else:  # UHF-FCI
            h1 = Ham.H1["cd"].copy()
            if Mu is not None:
                Mu_mat = np.eye(h1.shape[-1])
                nao = Mu_mat.shape[-1] // 2
                Mu_mat[range(nao), range(nao)] = -Mu
                Mu_mat[range(nao, nao * 2), range(nao, nao * 2)] = Mu
                mo_coeff = self.scfsolver.mf.mo_coeff[0]
                Mu_mat = mdot(mo_coeff.conj().T, Mu_mat, mo_coeff)
                h1[0] += Mu_mat

            Ham_restore = restore_Ham(Ham, 1, in_place=True)
            h2 = Ham_restore.H2["ccdd"]
            print("Ham shape: ", h2.shape)  # compact form in this case.
            self.fermion_hamil = self.construct_unrestricted_hamiltonian(h1, h2)

        self.n_qubits = 2 * Ham.H1["cd"].shape[-1]
        self.qubit_hamil = jordan_wigner(self.fermion_hamil)

        if self.n_qubits >= 12:
            pass
        else:  # for comparations
            mat = operator_to_matrix(self.n_qubits, self.qubit_hamil)
            value, _ = eigh(mat)
            # print("Fragment hartree Fock energy:", self.of_mole.hf_energy)
            print("***Directly diagonalization of fragment energy: ", value[0])

        print("Ham.norb is: ", Ham.norb)
        # In reality, we should record the index of the occupation number that equals to 1.
        print("Nelec is: ", self.nelec)

        terms = list(self.qubit_hamil.terms.keys())
        coefs = list(self.qubit_hamil.terms.values())
        # calculate the exp with the helper function
        print("********Test our custom hamiltonian in this situation.********")

        self.julia_hamil = Main.get_fermion_hamiltonian(self.n_qubits, terms, coefs)
        print("######Number of qubits######", self.n_qubits)
        self.n_orbitals = int(self.n_qubits / 2)


        # call the function similar to the get_fermion_hamiltonian in openfermion.
        # Then performs the quantum state preparation in the quantum circuit.
        # FIXME: to do construct the quantum solver for this system
        # Should generate a quantum computer solver to return such values
        # E, self.fcivec = self.cisolver.kernel(h1, h2, Ham.norb, self.nelec, \
        #                                       ci0=ci0, ecore=Ham.H0, pspace_size=pspace_size)

        if self.restricted:  # RHF-FCI
            if not self.opt_circuit: # no optimal circuit stored
                print("*******Call the Restricted_UCCSD_Simulate.******")
                E, self.opt_circuit, self.opt_param, self.para_key_list = self.Restricted_UCCSD_Simulate()
                print("Now the converge energy is: ", E)
            else:
                #call the optimized function will pre_optimized para as new input for the next iteration.
                print("Call the Restricted_UCCSD_Simulate with optimzed parameters initialize.")
                E, self.opt_circuit, self.opt_param, self.para_key_list = self.Restricted_UCCSD_Simulate_Optimal()
                print("*******Now the converge energy is: ", E)
                
        else:
            if not self.opt_circuit:
                print("Call the Unrestricted_UCCSD_Simulate.")
                E, self.opt_circuit, self.opt_param, self.para_key_list = self.Unrestricted_UCCSD_Simulate()
                print("Now the converge energy is: ", E)
            else:
                print("******Call the Unrestricted_UCCSD_Simulate with optimzed parameters initialize.******")
                E, self.opt_circuit, self.opt_param, self.para_key_list = self.Unrestricted_UCCSD_Simulate()
                print("Now the converge energy is: ", E)

        self.make_rdm1(Ham) # call this function to generate the onerdm_mo and then transformed back to AO.
        if Mu is not None:
            E -= np.einsum('pq, qp', Mu_mat, self.onepdm_mo[0])  # why only substract the index 0??
        if calc_rdm2:
            self.make_rdm2(Ham)

        self.optimized = True
        self.E = E
        # log.info("QC solver converged: %s", self.qc.converged)
        log.info("QC total energy: %s", self.E)
        print("QC total energy is: ", E)
        # D
        return self.onepdm, E # FIXME It seems in the AO basis for the loop while the Mo basis for the energy evaluations

    def construct_hamiltonian(self, one_body_integrals, two_body_integrals, threshold=1e-8):
        # Provide a unify interface
        if self.restricted:
            return self.construct_restricted_hamiltonian(one_body_integrals, two_body_integrals, threshold)
        else:
            return self.construct_unrestricted_hamiltonian(one_body_integrals, two_body_integrals, threshold)

    def get_onerdm(self):
        # Provide the unify interface for onerdm
        if self.restricted:
            return self.get_restricted_onerdm()
        else:
            return self.get_unrestricted_onerdm()

    def get_twordm(self):
        # Provide the unify interface for onerdm
        if self.restricted:
            return self.get_restricted_twordm()
        else:
            return self.get_unrestricted_twordm()
    def quantum_simulator(self):
        if self.restricted:
            return self.Restricted_UCCSD_Simulate()

        else:
            return self.Unrestricted_UCCSD_Simulate()



    def construct_unrestricted_hamiltonian(self, one_body_integrals, two_body_integrals, threshold=1e-8):
        """Construct the unrestricted Hamiltonian for this system.
        Here the one_body_integrals with dimensions(spin, norb, norb).
        and the two_body_integrals with dimensions(spin, norb, norb, norb, norb).
        Note be careful about the index order in this situations.
        h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # See PQRS convention in OpenFermion.hamiltonians.molecular_data
        """
        # This one_body_integral is with dimension as (spin, norb, norb)
        # This two_body_integral is with dimension as (spin, norb, norb, norb, norb)
        n_qubits = 2 * one_body_integrals.shape[-1]

        # h[p,q,r,s] = (ps|qr) = pyscf_eri[p,s,q,r]
        assert two_body_integrals.shape[0] == 3  # unrestriced
        two_body_integrals_new = np.zeros_like(two_body_integrals)
        for i in range(3):  # aa, bb, ab for this part
            # Actually this part is also unnecessary, if we always remember the map relationship.
            # Here I do this is to make it easily understand.
            two_body_integrals_new[i] = np.asarray(two_body_integrals[i].transpose(0, 2, 3, 1), order='C')

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros(
            (n_qubits, n_qubits, n_qubits, n_qubits))
        # Loop through integrals.
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):

                # Populate 1-body coefficients. Require p and q have same spin.
                # Note that we encode the alpha-> 2n, while beta-> 2n+1
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[0][p, q]
                one_body_coefficients[2 * p + 1, 2 * q +
                                      1] = one_body_integrals[1][p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):
                        # Mixed spin, note for this a, a, b, b
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                              s] = (two_body_integrals_new[2][p, q, r, s])
                        # b, b  and a, a, should change this accordingly
                        # two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                        #                       1] = (two_body_integrals[2][p, q, r, s])
                        # This the modification. FIXME: This should be careful.
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                              1] = (two_body_integrals_new[2][q, p, s, r])

                        # Same spin for alpha and beta respectively.
                        # Alpha spin
                        two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                              s] = (two_body_integrals_new[0][p, q, r, s])
                        # Beta spin
                        two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                              1, 2 * s +
                                              1] = (two_body_integrals_new[1][p, q, r, s])

        # Truncate. Set default value EQ_TOLERANCE = 1e-8
        one_body_coefficients[
            np.absolute(one_body_coefficients) < threshold] = 0.
        two_body_coefficients[
            np.absolute(two_body_coefficients) < threshold] = 0.

        # Ignore the constant part for now
        # For
        constant = 0.0
        # Also here should not have the prefactor 0.5, since all the coefficients are independently for alpha and beta

        molecular_hamiltonian = InteractionOperator(
            constant, one_body_coefficients,  1/2*two_body_coefficients) # 0.5 is not here compare to the restricted case.

        return molecular_hamiltonian

    def construct_restricted_hamiltonian(self, one_body_integrals, two_body_integrals, threshold=1e-8):
        """Construct the restricted Hamiltonian for this system.
        Note be careful about the index order in this situations.
        h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        # See PQRS convention in OpenFermion.hamiltonians.molecular_data
        """

        n_qubits = 2 * one_body_integrals.shape[0]

        # h[p,q,r,s] = (ps|qr) = pyscf_eri[p,s,q,r]
        two_body_integrals = np.asarray(two_body_integrals.transpose(0, 2, 3, 1), order='C')

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros(
            (n_qubits, n_qubits, n_qubits, n_qubits))
        # Loop through integrals.
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):

                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
                one_body_coefficients[2 * p + 1, 2 * q +
                                      1] = one_body_integrals[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):
                        # Mixed spin
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                              s] = (two_body_integrals[p, q, r, s])
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                              1] = (two_body_integrals[p, q, r, s])

                        # Same spin
                        two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                              s] = (two_body_integrals[p, q, r, s])
                        two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                              1, 2 * s +
                                              1] = (two_body_integrals[p, q, r, s])

        # Truncate. Set default value EQ_TOLERANCE = 1e-8
        one_body_coefficients[
            np.absolute(one_body_coefficients) < threshold] = 0.
        two_body_coefficients[
            np.absolute(two_body_coefficients) < threshold] = 0.

        # forget about the constant part for now
        constant = 0.0
        molecular_hamiltonian = InteractionOperator(
            constant, one_body_coefficients, 1 / 2 * two_body_coefficients)

        return molecular_hamiltonian


    def Unrestricted_UCCSD_Simulate(self):
        """This call the unrestricted unitary cc solver."""
        print("Begin the Unrestricted UCCSD Solver")
        begin = time.time()
        energy, opt_circuit, opt_param, para_key_list = Main.unrestricted_UCCSD_simulate_mol(self.n_qubits, self.n_electrons, self.julia_hamil, False)
        print("Now the Unrestricted UCCSD energy is: ", energy)
        print("End Unrestricted UCCSD Now")
        end = time.time()
        print("Time cost: ", end-begin)

        return energy, opt_circuit, opt_param, para_key_list
    
    def Unrestricted_UCCSD_Simulate_Optimal(self):
        """This call the unrestricted unitary cc solver."""
        print("Begin the optimized Unrestricted UCCSD Solver")
        begin = time.time()
        energy, opt_circuit, opt_param, para_key_list = Main.unrestricted_UCCSD_simulate_mol_optimal(self.n_qubits, self.n_electrons, self.julia_hamil, self.opt_param, self.para_key_list) # we can refactor this part.
        print("Now the optimized Unrestricted UCCSD energy is: ", energy)
        print("End optimized Unrestricted UCCSD Now")
        end = time.time()
        print("Time cost: ", end-begin)

        return energy, opt_circuit, opt_param, para_key_list

    def Restricted_UCCSD_Simulate_Optimal(self):
        """This call the restricted unitary cc solver."""
        print("Begin the optimal restricted UCCSD Solver")
        begin = time.time()
        energy, opt_circuit, opt_param, para_key_list = Main.restricted_UCCSD_simulate_mol_optimal(self.n_qubits, self.n_electrons, self.julia_hamil, self.opt_param, self.para_key_list)
        #Main.Restricted_UCCSD_simulate_mol(n_qubits,n_electrons_alpha_beta,hamil, real_mol=false)
        print("Now the optimal Restricted UCCSD energy is: ", energy)
        print("End optimal Restricted UCCSD Now")
        end = time.time()
        print("Time cost: ", end-begin)

        return energy, opt_circuit, opt_param, para_key_list
    
    def Restricted_UCCSD_Simulate(self):
        """This call the restricted unitary cc solver."""
        print("Begin the restricted UCCSD Solver")
        begin = time.time()
        energy, opt_circuit, opt_param, para_key_list = Main.restricted_UCCSD_simulate_mol(self.n_qubits, self.n_electrons, self.julia_hamil, False)
        print("Now the optimized Restricted UCCSD energy is: ", energy)
        print("End Restricted UCCSD Now")
        end = time.time()
        print("Time cost: ", end-begin)

        return energy, opt_circuit, opt_param, para_key_list


    def run_dmet_ham(self, Ham, last_aabb=True, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        """
        log.info("FCI solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff, compact=True, in_place=True)
        Ham = restore_Ham(Ham, 1, in_place=True)

        # calculate rdm2 in aa, bb, ab order
        self.make_rdm2(Ham)
        if Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape

            E1 = np.einsum('pq, qp', h1[0], r1[0]) * 2.0
            E2 = np.einsum('pqrs, pqrs', h2[0], r2[0]) * 0.5
        else:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape

            E1 = np.einsum('spq, sqp', h1, r1)
#             for item in permutations:

            E2_aa = 0.5 * np.einsum('pqrs, pqrs', h2[0], r2[0])
            E2_bb = 0.5 * np.einsum('pqrs, pqrs', h2[1], r2[1])
            E2_ab = np.einsum('pqrs, pqrs', h2[2], r2[2])

            # FIXME： Try the energy correction for this part due to the permutation of operator order
            E2 = E2_aa + E2_bb + E2_ab
            print("Run dmet Hamiltonian energy check: ", E2 + E1)
            print("One body energy is: ", E1)
            # Run the correction for the 1RDM
            # And the chemist notation of two body terms.

        E = E1 + E2
        E += Ham.H0
        log.debug(0, "run DMET Hamiltonian:\nE0 = %20.12f, E1 = %20.12f, "
                     "E2 = %20.12f, E = %20.12f", Ham.H0, E1, E2, E)
        return E

    def make_rdm1(self, Ham):
        log.debug(1, "Quantum Computing solver: solve rdm1")
        if Ham.restricted:
            # For restricted system, it is simple
            onepdm_mo = self.get_restricted_onerdm()
            self.onepdm_mo = (0.5*onepdm_mo)[np.newaxis]
            print("Finishing making restricted 1RDM.")
        else:
            self.onepdm_mo = self.get_unrestricted_onerdm()
            print("Finishing making unrestricted 1RDM.")

        # rotate back to the AO basis
        log.debug(1, "Quantum Computing solver: rotate rdm1 to AO")
        self.onepdm = transform_rdm1_to_ao_mol(self.onepdm_mo, \
                                               self.scfsolver.mf.mo_coeff)

    def make_rdm2(self, Ham, ao_repr=False):
        log.debug(1, "Quantum Computing solver: solve rdm2")
        if Ham.restricted:
            self.twopdm_mo = self.get_restricted_twordm()[np.newaxis]
            print("Finishing making restricted 2RDM.")
        else:
            self.twopdm_mo = self.get_unrestricted_twordm()
            self.twopdm_mo = self.twopdm_mo[[0,2,1]] # to make it the order as aa,ab,ba
            print("Finishing making 2RDM.")

        if ao_repr:
            log.debug(1, "Quantum Computing solver: rotate rdm2 to AO")
            self.twopdm = transform_rdm2_to_ao_mol(self.twopdm_mo, \
                                                   self.scfsolver.mf.mo_coeff)
        else:
            self.twopdm = None

        if not Ham.restricted and not self.ghf: # FIXME: IS this related to the gdf formats?
            self.twopdm_mo = self.twopdm_mo[[0, 2, 1]]
            if self.twopdm is not None:
                self.twopdm = self.twopdm[[0, 2, 1]]

    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm

    def get_onerdm(self):
        """Obtain the RDMs from the optimized amplitudes(circuit).
        Obtain the RDMs from the optimized amplitudes by using the
        same function for energy evaluation.
        The RDMs are computed by using each fermionic Hamiltonian term,
        transforming them and computing the elements one-by-one.
        Note that the Hamiltonian coefficients will not be multiplied
        as in the energy evaluation.
        The first element of the Hamiltonian is the nuclear repulsion
        energy term, not the Hamiltonian term.

        Returns:
            For unrestricted case, it is reture the current value for this situations
            (spin, numpy.array, numpy.array): One & two-particle RDMs

            For restrcited case, this is as usual.
            (numpy.array, numpy.array): One & two-particle RDMs

        """

        # Initialize the RDM arrays
        # Note this modification is to reuse the Hamiltonian for this part.
        # Note this part, we resue the code for the molecular system.
        begin = time.time()
        # check if this is a restricted system for this particularly problems
        if self.restricted:
            rdm1_np = np.zeros((self.n_orbitals,) * 2)
            one_rdm_op_list = {}
            two_rdm_op_list = {}
            # Loop over each element of Hamiltonian (non-zero value)
            for ikey, key in enumerate(self.fermion_hamil):
                key_length = len(key)
                # Treat one-body and two-body term accordingly
                if (key_length == 2):
                    pele, qele = int(key[0][0]), int(key[1][0])

                    iele, jele = pele // 2, qele // 2
                    # ap\dagger aq - aq\dagger ap

                    if (iele, jele) in one_rdm_op_list.keys():
                        one_rdm_op_list[(iele, jele)].append((pele, qele))
                    else:
                        one_rdm_op_list[(iele, jele)] = [(pele, qele)]  # count their appearance

            reg = Main.convert2statevec(self.n_qubits, self.opt_circuit)
            one_rdm_uniq_op = []
            for key, value in one_rdm_op_list.items():
                iele, jele = key
                rdm1_op = FermionOperator()
                for (pele, qele) in value:  # with spin information
                    rdm1_op += FermionOperator(((pele, 1), (qele, 0)))  # - FermionOperator(((qele,1),(pele,0)))
                qubit_hamiltonian2 = jordan_wigner(rdm1_op)
                # qubit_hamiltonian2.compress()
                if qubit_hamiltonian2 == QubitOperator():
                    continue  # do not record such values, I guess it will not enter into this branch
                else:
                    terms = list(qubit_hamiltonian2.terms.keys())
                    coefs = list(qubit_hamiltonian2.terms.values())
                    # print("terms and coefs: ", terms, coefs)
                    hamil = Main.get_fermion_hamiltonian(self.n_qubits, terms, coefs)
                    one_rdm_uniq_op.append((self.n_qubits, reg, iele, jele, hamil))

                # ***********************
            # deal with the one_body terms in this case with estimated information
            # one_two_rdm_op = one_rdm_uniq_op
            # one_body_estimation = []
            print("******Beginning only evaluate 1RDM ")  # main cost time is this part
            one_body_estimation = Main.parallel_estimate_energy(one_rdm_uniq_op)
            print("******Finishing only evaluate 1RDM ")
            for pair, opt_energy2 in one_body_estimation:
                # if len(pair)==2:
                (iele, jele) = pair
                rdm1_np[iele, jele] = rdm1_np[iele, jele] + np.real(opt_energy2)

        else:  # For unrestricted case
            print("Unrestricted case for the one-RDM.")
            one_rdm_aa_list = []
            one_rdm_bb_list = []
            reg = Main.convert2statevec(self.n_qubits, self.opt_circuit)
            # Loop over each element of Hamiltonian (non-zero value)
            for pele in range(self.n_orbitals):
                for qele in range(self.n_orbitals):
                    # aa
#                     rdm1_op = FermionOperator(((2*pele, 1), (2*qele, 0)))  #
                    rdm1_op = FermionOperator(((2*qele, 1), (2*pele, 0)))  #
                    qubit_hamiltonian2 = jordan_wigner(rdm1_op)
                    # qubit_hamiltonian2.compress()
                    if qubit_hamiltonian2 == QubitOperator():
                        continue  # do not record such values, I guess it will not enter into this branch
                    else:
                        terms = list(qubit_hamiltonian2.terms.keys())
                        coefs = list(qubit_hamiltonian2.terms.values())
                        # print("terms and coefs: ", terms, coefs)
                        if terms == [] or coefs == []:
                            continue
                        else:
                            hamil = Main.get_fermion_hamiltonian(self.n_qubits, terms, coefs)
                            one_rdm_aa_list.append((self.n_qubits, reg, pele, qele, hamil))
                    # bb
#                     dm1_op = FermionOperator(((2 * pele + 1, 1), (2 * qele+1, 0)))  #
                    dm1_op = FermionOperator(((2 * qele + 1, 1), (2 * pele+1, 0)))  #
                    qubit_hamiltonian2 = jordan_wigner(rdm1_op)
                    # qubit_hamiltonian2.compress()
                    if qubit_hamiltonian2 == QubitOperator():
                        continue  # do not record such values, I guess it will not enter into this branch
                    else:
                        terms = list(qubit_hamiltonian2.terms.keys())
                        coefs = list(qubit_hamiltonian2.terms.values())
                        # print("terms and coefs: ", terms, coefs)
                        if terms == [] or coefs == []:
                            continue
                        else:
                            hamil = Main.get_fermion_hamiltonian(self.n_qubits, terms, coefs)
                            one_rdm_bb_list.append((self.n_qubits, reg, pele, qele, hamil))


                # ***********************
            # deal with the one_body terms in this case with estimated information
            # one_two_rdm_op = one_rdm_uniq_op
            # one_body_estimation = []
            print("******Beginning only evaluate 1RDM ")  # main cost time is this part
            one_body_aa_estimation = Main.parallel_estimate_energy(one_rdm_aa_list)
            one_body_bb_estimation = Main.parallel_estimate_energy(one_rdm_bb_list)
            print("******Finishing only evaluate 1RDM ")
            rdm1_np_aa = np.zeros((self.n_orbitals, self.n_orbitals))
            rdm1_np_bb = np.zeros((self.n_orbitals, self.n_orbitals))
            rdm1_np = np.zeros((2, self.n_orbitals, self.n_orbitals))
            for pair, opt_energy2 in one_body_aa_estimation:
                # if len(pair)==2:
                (iele, jele) = pair
                rdm1_np_aa[iele, jele] = rdm1_np_aa[iele, jele] + np.real(opt_energy2)

            for pair, opt_energy2 in one_body_bb_estimation:
                # if len(pair)==2:
                (iele, jele) = pair
                rdm1_np_bb[iele, jele] = rdm1_np_bb[iele, jele] + np.real(opt_energy2)

            # Reshape the rdm1_np into alpha and beta parts
            rdm1_np = np.zeros((2, self.n_orbitals, self.n_orbitals))
            # alpha part
            rdm1_np[0] = rdm1_np_aa
            rdm1_np[1] = rdm1_np_bb


        end = time.time()
        print("parallel get only one rdm times: ", end - begin)

        return rdm1_np

    def get_unrestricted_onerdm(self):
        """Obtain the unrestricted oneRDMs from the optimized amplitudes.

        Obtain the RDMs from the optimized amplitudes by using the
        same function for energy evaluation.
        The RDMs are computed by using each fermionic Hamiltonian term,
        transforming them and computing the elements one-by-one.
        Note that the Hamiltonian coefficients will not be multiplied
        as in the energy evaluation.
        The first element of the Hamiltonian is the nuclear repulsion
        energy term, not the Hamiltonian term.
        Returns:
            (numpy.array, numpy.array): One RDMs(rdm1_np & rdm2_np, float64).
        """
        # Note this code is tested independently with the molecular system.
        print("Parallel get the one-rdm system")
        begin = time.time()
        n_qubits = self.n_qubits
        n_orbitals = n_qubits //2
        # Initialize the RDM arrays
        # Note this modification is to reuse the Hamiltonian for this part.
        rdm1_np=np.zeros((n_orbitals*2,)*2)
        rdm2_np=np.zeros((n_orbitals*2,)*4)
        one_rdm_op_list = {}
        two_rdm_op_list = {}
        # Loop over each element of Hamiltonian (non-zero value)
        for ikey,key in enumerate(self.fermion_hamil):
            key_length = len(key)
            # Treat one-body and two-body term accordingly
            if(key_length==2):
                pele, qele = int(key[0][0]), int(key[1][0])
                iele, jele = pele, qele
                # ap\dagger aq - aq\dagger ap
                if (iele,jele) in one_rdm_op_list.keys():
                    one_rdm_op_list[(iele,jele)].append((pele,qele))
                else:
                    one_rdm_op_list[(iele,jele)] = [(pele, qele)] #  count their appearance

        reg = Main.convert2statevec(n_qubits, self.opt_circuit)
        one_rdm_uniq_op= []
        for key, value in one_rdm_op_list.items():
            iele, jele = key
            rdm1_op = FermionOperator()
            for (pele, qele) in value: # with spin informations
                rdm1_op += FermionOperator(((pele,1),(qele,0))) #- FermionOperator(((qele,1),(pele,0)))
            qubit_hamiltonian2 = jordan_wigner(rdm1_op)
            qubit_hamiltonian2.compress()
            if qubit_hamiltonian2 == QubitOperator():
                continue # do not record such values
            else:
                terms= list(qubit_hamiltonian2.terms.keys())
                coefs= list(qubit_hamiltonian2.terms.values())
            # print("terms and coefs: ", terms, coefs)
                hamil = Main.get_fermion_hamiltonian(n_qubits, terms, coefs)
                one_rdm_uniq_op.append((n_qubits, reg, iele,jele,hamil))

        one_body_estimation = Main.parallel_estimate_energy(one_rdm_uniq_op)
        for pair, opt_energy2 in one_body_estimation:
            (iele, jele) = pair
            rdm1_np[iele,jele] += np.real(opt_energy2)


        onerdm = np.zeros((2, n_qubits//2, n_qubits//2))
        onerdm[0] = rdm1_np[0:n_qubits:2, 0:n_qubits:2]
        onerdm[1] = rdm1_np[1:n_qubits:2, 1:n_qubits:2]
        rdm1_np = None
        
        end = time.time()
        print("Finishing one-two rdm: ", end-begin)
        return onerdm

    def get_restricted_onerdm(self):
        """Obtain the restricted onerdm from the optimized amplitudes.

        Obtain the RDMs from the optimized amplitudes by using the
        same function for energy evaluation.
        The RDMs are computed by using each fermionic Hamiltonian term,
        transforming them and computing the elements one-by-one.
        Note that the Hamiltonian coefficients will not be multiplied
        as in the energy evaluation.
        The first element of the Hamiltonian is the nuclear repulsion
        energy term, not the Hamiltonian term.
        Returns:
            (numpy.array, numpy.array): One RDMs(rdm1_np & rdm2_np, float64).
        """
        # Note this code is tested independently with the molecular system.
        print("Parallel get the restricted onerdm.")
        begin = time.time()
        n_qubits = self.n_qubits
        n_orbitals = n_qubits //2
        # Initialize the RDM arrays
        # Note this modification is to reuse the Hamiltonian for this part.
        rdm1_np=np.zeros((n_orbitals,)*2)
        rdm2_np=np.zeros((n_orbitals,)*4)
        one_rdm_op_list = {}
        two_rdm_op_list = {}
        # Loop over each element of Hamiltonian (non-zero value)
        for ikey,key in enumerate(self.fermion_hamil):
            key_length = len(key)
            # Treat one-body and two-body term accordingly
            if(key_length==2):
                pele, qele = int(key[0][0]), int(key[1][0])
                iele, jele = pele//2, qele//2
                # ap\dagger aq - aq\dagger ap
                if (iele,jele) in one_rdm_op_list.keys():
                    one_rdm_op_list[(iele,jele)].append((pele,qele))
                else:
                    one_rdm_op_list[(iele,jele)] = [(pele, qele)] #  count their appearance

        reg = Main.convert2statevec(n_qubits, self.opt_circuit)
        one_rdm_uniq_op= []
        for key, value in one_rdm_op_list.items():
            iele, jele = key
            rdm1_op = FermionOperator()
            for (pele, qele) in value: # with spin informations
                rdm1_op += FermionOperator(((pele,1),(qele,0))) #- FermionOperator(((qele,1),(pele,0)))
            qubit_hamiltonian2 = jordan_wigner(rdm1_op)
            qubit_hamiltonian2.compress()
            if qubit_hamiltonian2 == QubitOperator():
                continue # do not record such values
            else:
                terms= list(qubit_hamiltonian2.terms.keys())
                coefs= list(qubit_hamiltonian2.terms.values())
            # print("terms and coefs: ", terms, coefs)
                hamil = Main.get_fermion_hamiltonian(n_qubits, terms, coefs)
                one_rdm_uniq_op.append((n_qubits, reg, iele,jele,hamil))


        one_body_estimation = Main.parallel_estimate_energy(one_rdm_uniq_op)
        for pair, opt_energy2 in one_body_estimation:
            (iele, jele) = pair
            rdm1_np[iele,jele] += np.real(opt_energy2)


        end = time.time()
        print("Finishing restricted one-two rdm: ", end-begin)
        return rdm1_np

    def get_unrestricted_twordm(self):
        """Obtain the unrestricted 2RDMs from the optimized amplitudes or quantum state.

        Obtain the RDMs from the optimized amplitudes by using the
        same function for energy evaluation.
        The RDMs are computed by using each fermionic Hamiltonian term,
        transforming them and computing the elements one-by-one.
        Note that the Hamiltonian coefficients will not be multiplied
        as in the energy evaluation.
        The first element of the Hamiltonian is the nuclear repulsion
        energy term, not the Hamiltonian term.

        Returns:
            (numpy.array, numpy.array): twordm with shape (aaaa, bbbb, aabb)

        """
        # Note this code is tested independently with the molecular system.
        print("Parallel get the unrestricted twordm system")
        begin = time.time()
        n_qubits = self.n_qubits
        n_orbitals = n_qubits //2

        rdm1_np=np.zeros((n_orbitals*2,)*2)
        rdm2_np=np.zeros((n_orbitals*2,)*4)
        one_rdm_op_list = {}
        two_rdm_op_list = {}
        # Loop over each element of Hamiltonian (non-zero value)
        for ikey,key in enumerate(self.fermion_hamil):
            key_length=len(key)
            # Treat one-body and two-body term accordingly
            if(key_length==4): #Seems this if is unnecessary
                pele, qele, rele, sele = int(key[0][0]), int(key[1][0]), int(key[2][0]), int(key[3][0])
                iele, jele, kele, lele = pele, qele, rele, sele

                if (iele,jele,kele,lele) in two_rdm_op_list.keys():
                    two_rdm_op_list[(iele,jele,kele,lele)].append((pele,qele,rele,sele)) # record the hamiltonian of that pair
                else:
                    two_rdm_op_list[(iele,jele,kele,lele)] = [(pele,qele,rele,sele)]

        reg = Main.convert2statevec(n_qubits, self.opt_circuit)
        one_rdm_uniq_op= []
        for key, value in one_rdm_op_list.items():
            iele, jele = key
            rdm1_op = FermionOperator()
            for (pele, qele) in value: # with spin informations
                rdm1_op += FermionOperator(((pele,1),(qele,0))) #- FermionOperator(((qele,1),(pele,0)))
            qubit_hamiltonian2 = jordan_wigner(rdm1_op)
            qubit_hamiltonian2.compress()
            if qubit_hamiltonian2 == QubitOperator():
                continue # do not record such values
            else:
                terms= list(qubit_hamiltonian2.terms.keys())
                coefs= list(qubit_hamiltonian2.terms.values())
            # print("terms and coefs: ", terms, coefs)
                hamil = Main.get_fermion_hamiltonian(n_qubits, terms, coefs)
                one_rdm_uniq_op.append((n_qubits, reg, iele,jele,hamil))

        two_rdm_uniq_op= []
        for key,value in two_rdm_op_list.items():
            iele, jele, kele, lele = key
            rdm2_op = FermionOperator()
            for (pele, qele, rele, sele) in value:
                # print("pele, qele, rele, sele: ", pele, qele, rele, sele)
                rdm2_op += FermionOperator(((pele,1),(qele,1),(rele,0),(sele,0))) # it is order by ap\dagger aq\dagger ar as
            # （0，3，1，2）
            qubit_hamiltonian2 = jordan_wigner(rdm2_op)
            qubit_hamiltonian2.compress()
            if qubit_hamiltonian2 == QubitOperator():
                continue # do not record such values
            else:
                terms= list(qubit_hamiltonian2.terms.keys())
                coefs= list(qubit_hamiltonian2.terms.values())
            # print("terms and coefs: ", terms, coefs)
                if terms !=[] and coefs !=[]:
                    hamil = Main.get_fermion_hamiltonian(n_qubits, terms, coefs)
                    two_rdm_uniq_op.append((n_qubits, reg,iele,jele,kele,lele,hamil))
            #***********************

        two_body_estimation = Main.parallel_estimate_energy(two_rdm_uniq_op)
        for pair, opt_energy2 in two_body_estimation:
            (iele, jele,kele,lele) = pair
            rdm2_np[iele,lele,jele,kele] += np.real(opt_energy2)


        twordm = np.zeros((3, n_qubits//2, n_qubits//2, n_qubits//2, n_qubits//2))
        twordm[0] = rdm2_np[0:n_qubits:2, 0:n_qubits:2, 0:n_qubits:2, 0:n_qubits:2] # aaaa
        twordm[1] = rdm2_np[1:n_qubits:2, 1:n_qubits:2, 1:n_qubits:2, 1:n_qubits:2] # bbbb
        twordm[2] = rdm2_np[0:n_qubits:2, 0:n_qubits:2, 1:n_qubits:2, 1:n_qubits:2] # aabb
        end = time.time()
        rdm2_np = None
        print("parallel finishing obtaining twordm: ", end-begin)
        return twordm


    def get_restricted_twordm(self):
        """Obtain the restricted 2RDMs from the optimized amplitudes or quantum state.

        Obtain the RDMs from the optimized amplitudes by using the
        same function for energy evaluation.
        The RDMs are computed by using each fermionic Hamiltonian term,
        transforming them and computing the elements one-by-one.
        Note that the Hamiltonian coefficients will not be multiplied
        as in the energy evaluation.
        The first element of the Hamiltonian is the nuclear repulsion
        energy term, not the Hamiltonian term.

        Returns:
            (numpy.array, numpy.array): twordm with shape (aaaa, bbbb, aabb)

        """
        # Note this code is tested independently with the molecular system.
        print("Parallel to obtain the restricted twordm.")
        begin = time.time()
        n_qubits = self.n_qubits
        n_orbitals = n_qubits //2

        rdm1_np=np.zeros((n_orbitals,)*2)
        rdm2_np=np.zeros((n_orbitals,)*4)
        one_rdm_op_list = {}
        two_rdm_op_list = {}
        # Loop over each element of Hamiltonian (non-zero value)
        for ikey,key in enumerate(self.fermion_hamil):
            key_length=len(key)
            # Treat one-body and two-body term accordingly
            if(key_length==4): #Seems this if is unnecessary
                pele, qele, rele, sele = int(key[0][0]), int(key[1][0]), int(key[2][0]), int(key[3][0])
                iele, jele, kele, lele = pele//2, qele//2, rele//2, sele//2

                if (iele,jele,kele,lele) in two_rdm_op_list.keys():
                    two_rdm_op_list[(iele,jele,kele,lele)].append((pele,qele,rele,sele)) # record the hamiltonian of that pair
                else:
                    two_rdm_op_list[(iele,jele,kele,lele)] = [(pele,qele,rele,sele)]

        reg = Main.convert2statevec(n_qubits, self.opt_circuit)
        
        two_rdm_uniq_op= []
        for key,value in two_rdm_op_list.items():
            iele, jele, kele, lele = key
            rdm2_op = FermionOperator()
            for (pele, qele, rele, sele) in value:
                # print("pele, qele, rele, sele: ", pele, qele, rele, sele)
                rdm2_op += FermionOperator(((pele,1),(qele,1),(rele,0),(sele,0))) # it is order by ap\dagger aq\dagger ar as
            # （0，3，1，2）
            qubit_hamiltonian2 = jordan_wigner(rdm2_op)
            qubit_hamiltonian2.compress()
            if qubit_hamiltonian2 == QubitOperator():
                continue # do not record such values
            else:
                terms= list(qubit_hamiltonian2.terms.keys())
                coefs= list(qubit_hamiltonian2.terms.values())
            # print("terms and coefs: ", terms, coefs)
                hamil = Main.get_fermion_hamiltonian(n_qubits, terms, coefs)
                two_rdm_uniq_op.append((n_qubits, reg,iele,jele,kele,lele,hamil))
            #***********************

        two_body_estimation = Main.parallel_estimate_energy(two_rdm_uniq_op)
        for pair, opt_energy2 in two_body_estimation:
            (iele, jele,kele,lele) = pair
            rdm2_np[iele,lele,jele,kele] += np.real(opt_energy2)


        end = time.time()
        print("Finishing restricted twordm: ", end-begin)
        return rdm2_np



    def cleanup(self):
        pass


