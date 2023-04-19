# Copyright (c) Bytedance Inc. 
# SPDX-License-Identifier: GPL-3.0-Only

import numpy as np
from openfermion.config import *
from openfermion.chem import MolecularData
from openfermion.transforms import jordan_wigner
from openfermionpyscf import run_pyscf
import itertools
from collections import OrderedDict as ordict
from openfermion import FermionOperator, jordan_wigner, normal_ordered
from projectq.ops import Rx, Ry, Rz, H, X
import openfermion as of
from itertools import product, combinations
import scipy
from collections import OrderedDict as ordict


def make_mol(bond_len):
    """
    Args: moleceluar bond length.
    Returns:  molecule(openfermion.hamiltonians.MolecularData):
    ## MolecularData for H2 with certain bond length, note this one is always an example for this case.
    """
    # load molecule
    atom_1 = 'H'
    atom_2 = 'Be'
    atom_3 = 'H'
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    coordinate_1 = (0.0, 0.0, 0.0)
    coordinate_2 = (0.0, 0.0, bond_len)
    coordinate_3 = (0.0, 0.0, 2*bond_len)
#     geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2)]
    geometry = [(atom_1, coordinate_1), (atom_2, coordinate_2), (atom_3, coordinate_3)]
    molecule = MolecularData(geometry, basis, multiplicity,
                             charge, description=str(bond_len))
    molecule = run_pyscf(molecule, run_scf=1, run_ccsd=1, run_fci=1)
    molecule.load()
    return molecule


def get_hamiltonian(molecule):
    """Get the coefficients and pauli terms of Hamiltonian of H_2 with given bond length.
    The return list of terms and coefficient can further be utilized to transform into Julia Yao type Hamiltonian.
    """
    qubit_hamiltonian = jordan_wigner(molecule.get_molecular_hamiltonian())
    qubit_hamiltonian.compress()
    return list(qubit_hamiltonian.terms.keys()), list(qubit_hamiltonian.terms.values())


def up_index(index):
    """ Represents the spin alpha part."""
    return 2*index


def down_index(index):
    """Represents the spin beta part."""
    return 2*index+1


def _transform2pauli(fermion_ansatz):
    """Transform a fermion ansatz to pauli ansatz based on jordan-wigner transformation."""
    out = ordict()
    for i in fermion_ansatz:
        qubit_generator = jordan_wigner(i[0])
        if qubit_generator.terms != {}:
            for key, term in qubit_generator.terms.items():
                if key not in out:
                    out[key] = ordict({i[1]: float(term.imag)})
                else:
                    if i[1] in out[key]:
                        out[key][i[1]] += float(term.imag)
                    else:
                        out[key][i[1]] = float(term.imag)
    return out


def decompose_single_term_time_evolution(term, para):
    """Decompose the this part into basic gates without considering the hardware constrains yet for now.
    And this decomposition is not consider hardware constraints"""
    if not isinstance(term, tuple):
        try:
            if len(term.terms) != 1:
                raise ValueError("Only work for single term time \
                    evolution operator, but get {}".format(len(term)))
            term = list(term.terms.keys())[0]
        except TypeError:
            raise Exception("Not supported type:{}".format(type(term)))

    out = []
    term = sorted(term)
    rxs = []
    if len(term) == 1:  # single pauli operator
        if term[0][1] == 'X':
            out.append(("Rx", para, term[0][0]))
        elif term[0][1] == 'Y':
            out.append(("Ry", para, term[0][0]))
        else:
            out.append(("Rz", para, term[0][0]))
    else:
        for index, action in term:
            if action == 'X':
                out.append(("H", index))
            elif action == 'Y':
                rxs.append(len(out))
                out.append(("Rx", np.pi / 2, index))

        for i in range(len(term) - 1):
            #out.append(X.on(term[i + 1][0], term[i][0]))
            # fist control, then target
            out.append(("CNOT", term[i][0], term[i+1][0]))
        out.append(("Rz", {i: 2 * j for i, j in para.items()}, term[-1][0]))
        for i in range(len(out) - 1)[::-1]:
            if i in rxs:
                # deal with Ry with different phase
                out.append(("Rx", np.pi * 3.5, out[i][2]))
            else:
                out.append(out[i])
    return out


def _pauli2circuit(pauli_ansatz):
    """Transform a pauli ansatz to parameterized quantum circuit."""
    circuit = []
    for k, v in pauli_ansatz.items():
        circuit += (decompose_single_term_time_evolution(k, v))
    return circuit


def para_kpuccgsd_generator(n_qubits, n_electrons, th=-1, k=1):
    # This set up default by up and down then up and down pattern
    n_qubits = n_qubits
    n_electrons = n_electrons
    params = {}
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    k_out = []
    out_tmp = []
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied
    cnt = 0

    # Unpack amplitudes
    #n_single_amplitudes = n_occupied * n_virtual
    n_single_amplitudes = n_spatial_orbitals * \
        (n_spatial_orbitals-1)  # for the generalized kpUCCGSD
    # Generate excitations
    spin_index_functions = [up_index, down_index]
    # Generate all spin-conserving single and double excitations derived
    # from one spatial occupied-virtual pair
    index = 0
    for it in range(k):
        #         print("it is: ", it)
        out = []
        offset = it*2*n_single_amplitudes
        for i, (p, q) in enumerate(
                itertools.product(range(n_spatial_orbitals), range(n_spatial_orbitals))):  # A modified acutally not generatlized version
            # p and q should not be same
            if p == q:
                pass
            # Get indices of spatial orbitals
#             virtual_spatial = n_occupied + p
            else:

                virtual_spatial = p
                occupied_spatial = q
                virtual_up = virtual_spatial * 2
                occupied_up = occupied_spatial * 2
                virtual_down = virtual_spatial * 2 + 1
                occupied_down = occupied_spatial * 2 + 1

                single_amps = 1e-5
                double1_amps = 1e-5
                single_amps_name = 'p' + str(index + offset)
                double1_amps_name = 'p' + \
                    str(index + n_single_amplitudes+offset)
                index += 1

                # deal with the spin up/down single excitation part
                cnt += 1  # only counts the number
                #
                params[single_amps_name] = single_amps
                fermion_ops1 = FermionOperator(
                    ((virtual_up, 1), (occupied_up, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_up, 1), (virtual_up, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, single_amps_name])

                fermion_ops1 = FermionOperator(
                    ((virtual_down, 1), (occupied_down, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_down, 1), (virtual_down, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, single_amps_name])

                # Generate pair-double excitation
#                     if abs(double1_amps) > th:
                cnt += 1
                params[double1_amps_name] = double1_amps
                fermion_ops1 = FermionOperator(
                    ((virtual_up, 1), (occupied_up, 0), (virtual_down, 1),
                     (occupied_down, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_down, 1), (virtual_down, 0),
                     (occupied_up, 1), (virtual_up, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, double1_amps_name])

                #print("cnt is: ", cnt)
    return k_out, params


def _para_unrestricted_uccsd_generator(n_qubits, n_electrons_list, th=-1):
    """"Construct the unrestricted ccsd for the unrestricted coupled cluster.
    Also for this one, we utilize this mainly for embedded system. But it can also use for the generally
    unrestricted system, since we set the initial parameters to be very small."""
    params = {}
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')
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

    n_double2_alpha_beta = int(
        n_occ_beta*(n_occ_beta-1)*n_virt_beta*(n_virt_beta-1))

    return out, params


def para_kpuccgsd_generator(n_qubits, n_electrons, th=-1, k=1):
    """This set up default by up and down then up and down pattern"""
    n_qubits = n_qubits
    n_electrons = n_electrons
    params = {}
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    k_out = []
    out_tmp = []
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied
    cnt = 0

    # Unpack amplitudes
    #n_single_amplitudes = n_occupied * n_virtual
    n_single_amplitudes = n_spatial_orbitals * \
        (n_spatial_orbitals-1)  # for the generalized kpUCCGSD
    # Generate excitations
    spin_index_functions = [up_index, down_index]
    # Generate all spin-conserving single and double excitations derived
    # from one spatial occupied-virtual pair
    index = 0
    for it in range(k):
        #         print("it is: ", it)
        out = []
        offset = it*2*n_single_amplitudes
        for i, (p, q) in enumerate(
                itertools.product(range(n_spatial_orbitals), range(n_spatial_orbitals))):  # A modified acutally not generatlized version
            # p and q should not be same
            if p == q:
                pass
            # Get indices of spatial orbitals
#             virtual_spatial = n_occupied + p
            else:

                virtual_spatial = p
                occupied_spatial = q
                virtual_up = virtual_spatial * 2
                occupied_up = occupied_spatial * 2
                virtual_down = virtual_spatial * 2 + 1
                occupied_down = occupied_spatial * 2 + 1

                single_amps = 1e-5
                double1_amps = 1e-5
                single_amps_name = 'p' + str(index + offset)
                double1_amps_name = 'p' + \
                    str(index + n_single_amplitudes+offset)
                index += 1

                # deal with the spin up/down single excitation part
                cnt += 1  # only counts the number
                #
                params[single_amps_name] = single_amps
                fermion_ops1 = FermionOperator(
                    ((virtual_up, 1), (occupied_up, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_up, 1), (virtual_up, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, single_amps_name])

                fermion_ops1 = FermionOperator(
                    ((virtual_down, 1), (occupied_down, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_down, 1), (virtual_down, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, single_amps_name])

                # Generate pair-double excitation
#                     if abs(double1_amps) > th:
                cnt += 1
                params[double1_amps_name] = double1_amps
                fermion_ops1 = FermionOperator(
                    ((virtual_up, 1), (occupied_up, 0), (virtual_down, 1),
                     (occupied_down, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_down, 1), (virtual_down, 0),
                     (occupied_up, 1), (virtual_up, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, double1_amps_name])
                #print("cnt is: ", cnt)

        k_out.append(out)

    return k_out, params


def _para_uccsd_singlet_generator(n_qubits, n_electrons, th=-1):
    #n_qubits = mol.n_qubits
    #n_electrons = mol.n_electrons
    params = {}
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')
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
    return out, params


def _para_guccsd_generator(n_qubits, n_electrons, th=-1):
    params = {}
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')
    out = []
    out_tmp = []
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Unpack amplitudes
    # n_single_amplitudes = n_occupied * n_virtual
    n_single_amplitudes = n_spatial_orbitals*(n_spatial_orbitals-1)

    # Generate excitations
    spin_index_functions = [up_index, down_index]
    # Generate all spin-conserving single and double excitations derived
    # from one spatial occupied-virtual pair
    index = 0
    for i, (p, q) in enumerate(
            itertools.product(range(n_spatial_orbitals), range(n_spatial_orbitals))):

        # Get indices of spatial orbitals
        if p != q:
            virtual_spatial = p
            occupied_spatial = q
            virtual_up = virtual_spatial * 2
            occupied_up = occupied_spatial * 2
            virtual_down = virtual_spatial * 2 + 1
            occupied_down = occupied_spatial * 2 + 1

            single_amps = 1e-5
            double1_amps = 1e-5
            single_amps_name = 'p' + str(index)
            #double1_amps_name = 'p' + str(index + n_single_amplitudes)
            double1_amps_name = 'p' + str(index + 1)
            index += 2

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

    out.extend(out_tmp)
    out_tmp = []
    # Generate all spin-conserving double excitations derived
    # from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
            itertools.combinations(
                itertools.product(range(n_spatial_orbitals), range(n_spatial_orbitals)), 2)):

        if ((p != q) and (r != s)):
            # Get indices of spatial orbitals
            #print("Now p,q,r,s is: ", p,q,r,s)
            virtual_spatial_1 = p
            occupied_spatial_1 = q
            virtual_spatial_2 = r
            occupied_spatial_2 = s

            virtual_1_up = virtual_spatial_1 * 2
            occupied_1_up = occupied_spatial_1 * 2
            virtual_2_down = virtual_spatial_2 * 2 + 1
            occupied_2_down = occupied_spatial_2 * 2 + 1

            double2_amps = 1e-5
            double2_amps_name = 'p' + str(index)
            index += 1
            if abs(double2_amps) > th:
                params[double2_amps_name] = double2_amps
                fermion_ops1 = FermionOperator(
                    ((virtual_1_up, 1), (occupied_1_up, 0), (virtual_2_down, 1),
                     (occupied_2_down, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_2_down, 1), (virtual_2_down, 0), (occupied_1_up, 1),
                     (virtual_1_up, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, double2_amps_name])

                fermion_ops1 = FermionOperator(
                    ((virtual_1_up+1, 1), (occupied_1_up+1, 0), (virtual_2_down-1, 1),
                     (occupied_2_down-1, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((occupied_2_down-1, 1), (virtual_2_down-1, 0), (occupied_1_up+1, 1),
                     (virtual_1_up+1, 0)), 1)
                out.append([fermion_ops1 - fermion_ops2, double2_amps_name])

                # add its pair terms
    return out, params


def construct_kp_uccgsd(n_qubits, n_electrons,  th=-1, k=1):
    """Construct the kgeneralized uccgsd operator pool in this case."""
    k_out, params = para_kpuccgsd_generator(n_qubits, n_electrons, th, k)
    cir = []
    for item in k_out:
        pauli_ansatz = _transform2pauli(item)
        circuit = _pauli2circuit(pauli_ansatz)
        cir.extend(circuit)
    return cir, params


# it benifit both uccsd and guccsd.
def construct_uccsd(n_qubits, n_electrons,  th=-1):
    #kout, params = _para_uccsd_singlet_generator(n_qubits, n_electrons, th)
    kout, params = _para_guccsd_generator(n_qubits, n_electrons, -1)
    pauli_ansatz = _transform2pauli(kout)
    circuit = _pauli2circuit(pauli_ansatz)
    return circuit, params


# unrestricted unitary ccsd related quantum gates
def construct_unrestricted_uccsd(n_qubits, n_electrons_alpha_beta,  th=-1):
    out, params = _para_unrestricted_uccsd_generator(
        n_qubits, n_electrons_alpha_beta, th=-1)
    pauli_ansatz = _transform2pauli(out)
    circuit = _pauli2circuit(pauli_ansatz)
    return circuit, params


def wrap_minimizer(simulate, params_list, args=None, method="L-BFGS-B", jac=True):
    # TO support change the setting of the configuration in Dict.
    myfactr = 1e2
    # r = scipy.optimize.minimize(, options={'ftol' : myfactr * np.finfo(float).eps)
    res = scipy.optimize.minimize(simulate, params_list, args=args, method=method, jac=jac, options={
                                  'ftol': myfactr*np.finfo(float).eps})
    return res


def spatial2spin_idx(idx, spin):
    """Convert the spatial index to spin index.
    with alpha -> 2*alpha + 0, beta -> 2*beta+1. Basically 2*idx + spin in this case. 
    (This for python, where index start with zero.)
    """
    return 2*idx + spin


def single_excitation_operator_restricted(n_virtual, n_occupied, out, params, params_pair_map, th=-1, offset=0):
    """This is assumes the alpha beta alpha beta order. 
    Note all the initial parameters are set to be the same as 1e-5. This could be changed."""
    for i, (p, q) in enumerate(product(range(n_virtual), range(n_occupied))):
        # Get indices of spatial orbitals
        virt_spa_idx = n_occupied + p
        occ_spa_idx = q
        single_amps = 1.0e-5
        single_amps_name = 'p' + str(i + offset)
        for spin in range(2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            virt_spin_idx = spatial2spin_idx(virt_spa_idx, spin)
            occ_spin_idx = spatial2spin_idx(occ_spa_idx, spin)
            # Generate single excitations
            if abs(single_amps) > th:
                params[single_amps_name] = single_amps
                fermion_ops1 = FermionOperator(
                    ((occ_spin_idx, 1), (virt_spin_idx, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((virt_spin_idx, 1), (occ_spin_idx, 0)), 1)
                op = normal_ordered(fermion_ops1 - fermion_ops2)
                if op != FermionOperator():
                    out.append([op, single_amps_name])
                    #out.append([fermion_ops1 - fermion_ops2, single_amps_name])
                    if single_amps_name in params_pair_map.keys():
                        params_pair_map[single_amps_name].append(
                            (occ_spin_idx, virt_spin_idx))
                    else:
                        params_pair_map[single_amps_name] = [
                            (occ_spin_idx, virt_spin_idx)]


def single_excitation_operator_unrestricted(n_virtual, n_occupied, out, params, params_pair_map, th=-1, offset=0):
    # This is assumes the alpha beta alpha beta order.
    # Note all the initial parameters are set to be the same as 1e-5. This could be changed.
    for i, (p, q) in enumerate(product(range(n_virtual), range(n_occupied))):
        # Get indices of spatial orbitals
        virt_spa_idx = n_occupied + p
        occ_spa_idx = q
        single_amps = 1.0e-5
        single_amps_name = 'p' + str(i + offset)
        term = FermionOperator()
        for spin in range(2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            virt_spin_idx = spatial2spin_idx(virt_spa_idx, spin)
            occ_spin_idx = spatial2spin_idx(occ_spa_idx, spin)
            # Generate single excitations
            if abs(single_amps) > th:
                params[single_amps_name] = single_amps
                fermion_ops1 = FermionOperator(
                    ((occ_spin_idx, 1), (virt_spin_idx, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((virt_spin_idx, 1), (occ_spin_idx, 0)), 1)
                op = normal_ordered(fermion_ops1 - fermion_ops2)
                term += op

        if term != FermionOperator():
            out.append([term, single_amps_name])
            #out.append([fermion_ops1 - fermion_ops2, single_amps_name])
            if single_amps_name in params_pair_map.keys():
                params_pair_map[single_amps_name].append(
                    (occ_spin_idx, virt_spin_idx))
            else:
                params_pair_map[single_amps_name] = [
                    (occ_spin_idx, virt_spin_idx)]


def bonsonic_double_excitation_operator(n_virtual, n_occupied, out, params, params_pair_map, th=-1, offset=0):
    """This excitation corresponding to the two electron excitation 
    which couple to two spatial orbitals. Basically a sptail orbital excited to 
    another spatial orbital with two electrons at the same time.
    Note all the initial parameters are set to be the same as 1e-5. This could be changed."""
    for i, (p, q) in enumerate(product(range(n_virtual), range(n_occupied))):
        # Get indices of spatial orbitals
        virt_spa_idx = n_occupied + p
        occ_spa_idx = q
        double_amps = 1.0e-5
        double_amps_name = 'p' + str(i+offset)
        term = FermionOperator()
        for spin in range(2):
            virt_spin_alpha = spatial2spin_idx(virt_spa_idx, spin)
            occ_spin_alpha = spatial2spin_idx(occ_spa_idx, spin)
            virt_spin_beta = spatial2spin_idx(virt_spa_idx, 1-spin)
            occ_spin_beta = spatial2spin_idx(occ_spa_idx, 1-spin)

            # Generate single excitations
            if abs(double_amps) > th:
                params[double_amps_name] = double_amps
                fermion_ops1 = FermionOperator(
                    ((occ_spin_alpha, 1), (virt_spin_alpha, 0), (occ_spin_beta, 1), (virt_spin_beta, 0)), 1)
                fermion_ops2 = FermionOperator(
                    ((virt_spin_beta, 1), (occ_spin_beta, 0), (virt_spin_alpha, 1), (occ_spin_alpha, 0)), 1)
                op = normal_ordered(fermion_ops1 - fermion_ops2)
                term += op

        if term != FermionOperator():
            out.append([term, double_amps_name])
            if double_amps_name in params_pair_map.keys():
                params_pair_map[double_amps_name].append(
                    (occ_spin_alpha, virt_spin_alpha, occ_spin_beta, virt_spin_beta))
            else:
                params_pair_map[double_amps_name] = [
                    (occ_spin_alpha, virt_spin_alpha, occ_spin_beta, virt_spin_beta)]


def non_bonsonic_double_excitation_operator(n_virtual, n_occupied, out, params, params_pair_map, th=-1, offset=0):
    """This excitation corresponding to the two electron excitation which couple more than spatial orbitals."""
    for i, ((p, q), (r, s)) in enumerate(combinations(
            product(range(n_virtual), range(n_occupied)), 2)):
        virt_spa1_idx = n_occupied + p
        occ_spa1_idx = q
        virt_spa2_idx = n_occupied + r
        occ_spa2_idx = s

        double2_amps = 1e-5
        double2_amps_name = 'p' + str(i+offset)

        # Generate double excitations
        for (spin_a, spin_b) in product(range(2), repeat=2):
            # Get indices of spin orbitals
            virtual_1_a = spatial2spin_idx(virt_spa1_idx, spin_a)
            occupied_1_a = spatial2spin_idx(occ_spa1_idx, spin_a)
            virtual_2_b = spatial2spin_idx(virt_spa2_idx, spin_b)
            occupied_2_b = spatial2spin_idx(occ_spa2_idx, spin_b)
            if abs(double2_amps) > th:
                # note for this, we change the coefficient to 0.5
                params[double2_amps_name] = double2_amps
                fermion_ops1 = FermionOperator(
                    ((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1),
                     (occupied_2_b, 0)), 0.5)
                fermion_ops2 = FermionOperator(
                    ((occupied_2_b, 1), (virtual_2_b, 0), (occupied_1_a, 1),
                     (virtual_1_a, 0)), 0.5)
                op = normal_ordered(fermion_ops1 - fermion_ops2)
                if op != FermionOperator():
                    out.append([op, double2_amps_name])
                #out.append([fermion_ops1 - fermion_ops2, double2_amps_name])
                    if double2_amps_name in params_pair_map.keys():
                        params_pair_map[double2_amps_name].append(
                            (virtual_1_a, occupied_1_a, virtual_2_b, occupied_2_b))
                    else:
                        params_pair_map[double2_amps_name] = [
                            (virtual_1_a, occupied_1_a, virtual_2_b, occupied_2_b)]


def non_bonsonic_double_excitation_operator_restricted(n_virtual, n_occupied, out, params, params_pair_map, th=-1, offset=0):
    """This excitation corresponding to the two electron excitation, which couple more than spatial orbitals."""
    for i, ((p, q), (r, s)) in enumerate(combinations(
            product(range(n_virtual), range(n_occupied)), 2)):
        virt_spa1_idx = n_occupied + p
        occ_spa1_idx = q
        virt_spa2_idx = n_occupied + r
        occ_spa2_idx = s

        double2_amps = 1e-5
        double2_amps_name = 'p' + str(i+offset)

        # Generate double excitations
        term = FermionOperator()
        for (spin_a, spin_b) in product(range(2), repeat=2):
            # Get indices of spin orbitals
            virtual_1_a = spatial2spin_idx(virt_spa1_idx, spin_a)
            occupied_1_a = spatial2spin_idx(occ_spa1_idx, spin_a)
            virtual_2_b = spatial2spin_idx(virt_spa2_idx, spin_b)
            occupied_2_b = spatial2spin_idx(occ_spa2_idx, spin_b)
            if abs(double2_amps) > th:
                # note for this, we change the coefficient to 0.5
                params[double2_amps_name] = double2_amps
                fermion_ops1 = FermionOperator(
                    ((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1),
                     (occupied_2_b, 0)), 0.5)
                fermion_ops2 = FermionOperator(
                    ((occupied_2_b, 1), (virtual_2_b, 0), (occupied_1_a, 1),
                     (virtual_1_a, 0)), 0.5)
                op = normal_ordered(fermion_ops1 - fermion_ops2)
                term += op
        if term != FermionOperator():
            out.append([term, double2_amps_name])
        #out.append([fermion_ops1 - fermion_ops2, double2_amps_name])
            if double2_amps_name in params_pair_map.keys():
                params_pair_map[double2_amps_name].append(
                    (virtual_1_a, occupied_1_a, virtual_2_b, occupied_2_b))
            else:
                params_pair_map[double2_amps_name] = [
                    (virtual_1_a, occupied_1_a, virtual_2_b, occupied_2_b)]


def _para_uccsd_generator_pool_new(n_qubits, n_electrons, th=-1):
    """Generate the corresponding operator pool for uccsd and its variants as we like."""
    params = {}
    params_pair_map = {}
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')
    out = []
    out_tmp = []
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Unpack amplitudes
    n_single_amplitudes = n_occupied * n_virtual
    out = []
    params = {}
    params_pair_map = {}
    single_excitation_operator_restricted(
        n_virtual, n_occupied, out, params, params_pair_map, th, 0)
    bonsonic_double_excitation_operator(
        n_virtual, n_occupied, out, params, params_pair_map, th, n_single_amplitudes)
    non_bonsonic_double_excitation_operator(
        n_virtual, n_occupied, out, params, params_pair_map, th, 2*n_single_amplitudes)

    return out, params, params_pair_map


def _para_uccsd_generator_pool_new2(n_qubits, n_electrons, th=-1):
    """Generate the corresponding operator pool for uccsd and its variants as we like."""
    params = {}
    params_pair_map = {}
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')
    out = []
    out_tmp = []
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Unpack amplitudes
    n_single_amplitudes = n_occupied * n_virtual
    out = []
    params = {}
    params_pair_map = {}
    single_excitation_operator_restricted(
        n_virtual, n_occupied, out, params, params_pair_map, th, 0)
    bonsonic_double_excitation_operator(
        n_virtual, n_occupied, out, params, params_pair_map, th, n_single_amplitudes)
    non_bonsonic_double_excitation_operator_restricted(
        n_virtual, n_occupied, out, params, params_pair_map, th, 2*n_single_amplitudes)

    return out, params, params_pair_map


# This could be further simplified.
def adapt_gradient_op_pool_list(fermion_ansatz):
    """Change the output of the pool into the corresponding Qubit Operator for further test."""
    op_ls = []
    for op in fermion_ansatz:
        qubit_op = jordan_wigner(op)
        terms = list(qubit_op.terms.keys())
        coefs = list(qubit_op.terms.values())
        op_ls.append((terms, coefs))
    return op_ls


def _transform2pauli_onebyone(fermion_ansatz):
    """Transform a fermion ansatz to pauli ansatz based on jordan-wigner
    transformation."""
    all_out = []
    for i in fermion_ansatz:
        out = ordict()
        qubit_generator = jordan_wigner(i[0])
        if qubit_generator.terms != {}:
            for key, term in qubit_generator.terms.items():
                if key not in out:
                    out[key] = ordict({i[1]: float(term.imag)})
                else:
                    if i[1] in out[key]:
                        out[key][i[1]] += float(term.imag)
                    else:
                        out[key][i[1]] = float(term.imag)
        all_out.append(out)
    return all_out


def decompose_single_term_time_evolution(term, para):
    """Decompose the exponetial single Pauli operator into a sequence of single qubit and two-qubit gates."""
    if not isinstance(term, tuple):
        try:
            if len(term.terms) != 1:
                raise ValueError("Only work for single term time \
                    evolution operator, but get {}".format(len(term)))
            term = list(term.terms.keys())[0]
        except TypeError:
            raise Exception("Not supported type:{}".format(type(term)))

    out = []
    term = sorted(term)
    rxs = []
    if len(term) == 1:  # single pauli operator
        if term[0][1] == 'X':
            out.append(("Rx", para, term[0][0]))
        elif term[0][1] == 'Y':
            out.append(("Ry", para, term[0][0]))
        else:
            out.append(("Rz", para, term[0][0]))
    else:
        for index, action in term:
            if action == 'X':
                out.append(("H", index))
            elif action == 'Y':
                rxs.append(len(out))
                out.append(("Rx", np.pi / 2, index))

        for i in range(len(term) - 1):
            #out.append(X.on(term[i + 1][0], term[i][0]))
            # fist control, then target
            out.append(("CNOT", term[i][0], term[i+1][0]))
        out.append(("Rz", {i: 2 * j for i, j in para.items()}, term[-1][0]))
        for i in range(len(out) - 1)[::-1]:
            if i in rxs:
                # deal with Ry with different phase
                out.append(("Rx", np.pi * 3.5, out[i][2]))
            else:
                out.append(out[i])
    return out


def _pauli2circuit_onebyone(pauli_ansatz_ls):
    """Transform the pauli ansatz into a series of gate for gate sequence
    that gradually increasing the operator such as the ADAPT VQE in this case."""
    circuit_ls = []
    for pauli_ansatz in pauli_ansatz_ls:
        # each operator corresponding to the corresponding quantum circuit.
        circuit = []
        for term, coeff in pauli_ansatz.items():
            circuit += (decompose_single_term_time_evolution(term, coeff))
        circuit_ls.append(circuit)
    return circuit_ls


if __name__ == "__main__":
    bond_len = 1.0
    # Test for the LiH moleculue in this case.
    mol = make_mol(bond_len)
    n_qubits = mol.n_qubits
    n_electrons = mol.n_electrons
    print("Now n_qubit is: ", n_qubits)
    print("Now the electron is: ", n_electrons)
