# Copyright (c) Bytedance Inc. 
# SPDX-License-Identifier: GPL-3.0-Only

import pyscf
from pyscf import ao2mo
from functools import reduce
from openfermion import InteractionOperator, FermionOperator, QubitOperator, jordan_wigner
import numpy as np
import time

def up_index(index):
    """For spin alpha."""
    return 2 * index


def down_index(index):
    """For spin beta."""
    return 2 * index + 1

def spinorb_from_spatial_unrestricted(one_body_integrals, two_body_integrals, threshold=1e-8):
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
    # I have change the Hamiltonian in this case.

    # h[p,q,r,s] = (ps|qr) = pyscf_eri[p,s,q,r]
    assert two_body_integrals.shape[0] == 3  # unrestricted
    two_body_integrals_new = np.zeros_like(two_body_integrals)
    for i in range(3):  # aa, bb, ab for this part
        # I make a mistake in this case.
        two_body_integrals_new[i] = np.asarray(two_body_integrals[i].transpose(0, 2, 3, 1), order='C')

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):
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

    return one_body_coefficients, two_body_coefficients

def spinorb_from_spatial_restricted(one_body_integrals, two_body_integrals, threshold=1e-8):
    """Construct the unrestricted Hamiltonian for this system.
    Here the one_body_integrals with dimensions(spin, norb, norb).
    and the two_body_integrals with dimensions(spin, norb, norb, norb, norb).
    Note be careful about the index order in this situations.
    h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
    # See PQRS convention in OpenFermion.hamiltonians.molecular_data
    """
    # This one_body_integral is with dimension as (spin, norb, norb)
    # This two_body_integral is with dimension as (spin, norb, norb, norb, norb)
    # Note this is also for the integral directly from pyscf with format, <pq|rs> are the corresponding pairs
    n_qubits = 2 * one_body_integrals.shape[-1]
    # I have change the Hamiltonian in this case.

    # h[p,q,r,s] = (ps|qr) = pyscf_eri[p,s,q,r]
    two_body_integrals_new = np.zeros_like(two_body_integrals)

    two_body_integrals_new = np.asarray(two_body_integrals.transpose(0, 2, 3, 1), order='C')

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):
            # Note that we encode the alpha-> 2n, while beta-> 2n+1
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
                    # Mixed spin, note for this a, a, b, b
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (two_body_integrals_new[p, q, r, s])
                    # b, b  and a, a, should change this accordingly
                    # two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                    #                       1] = (two_body_integrals[2][p, q, r, s])
                    # This the modification. FIXME: This should be careful.
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (two_body_integrals_new[p, q, r, s])

                    # Same spin for alpha and beta respectively.
                    # Alpha spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (two_body_integrals_new[p, q, r, s])
                    # Beta spin
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s +
                                          1] = (two_body_integrals_new[p, q, r, s])

    # Truncate. Set default value EQ_TOLERANCE = 1e-8
    one_body_coefficients[
        np.absolute(one_body_coefficients) < threshold] = 0.
    two_body_coefficients[
        np.absolute(two_body_coefficients) < threshold] = 0.

    return one_body_coefficients, two_body_coefficients


def mdot(*args):
    """
    Reduced matrix dot.
    """
    return reduce(np.dot, args)


def ao2mo_ham(mf_mol, restricted=False, compact=True, restore=True):
    """Generate Hamiltonian for unrestricted system and test it in the quantum computer solvers.
    Restore it to four rank tensor (norb, norb, norb, norb)
    """
    h1e_ao = mf_mol.get_hcore()
    eri_ao = mf_mol._eri
    C = mf_mol.mo_coeff
    norb = h1e_ao.shape[-1]

    # Change it into the MO basis for this particularly case
    if restricted:
        h1e = mdot(C[0].conj().T, h1e_ao, C[0])[np.newaxis]
        eri = ao2mo.restore(1, eri_ao, norb)
        eri = ao2mo.full(eri, C[0], compact=compact)[np.newaxis]
        if restore:
            eri = ao2mo.restore(1, eri, norb)
        return h1e, eri

    else:  # Unrestricted case
        norb_pair = norb * (norb + 1) // 2
        h1e = np.zeros((2, norb, norb))
        for s in range(2):
            h1e[s] = mdot(C[s].conj().T, h1e_ao, C[s])
        # H2
        if compact:
            eri = np.zeros((3, norb_pair, norb_pair))  # norb_pair = norb * (norb + 1) // 2
        else:
            eri = np.zeros((3, norb * norb, norb * norb))

        eri_aa = ao2mo.restore(8, eri_ao, norb)
        eri[0] = ao2mo.full(eri_aa, C[0], compact=compact)
        print("Eri0 shape: ", eri[0].shape)
        eri_aa = None

        eri_bb = ao2mo.restore(8, eri_ao, norb)
        eri[1] = ao2mo.full(eri_bb, C[1], compact=compact)
        eri_bb = None

        eri_ab = ao2mo.restore(4, eri_ao, norb)
        # alpha_create alpha_create beta_des beta_des
        print("Eri_ab: ", eri_ab.shape)
        eri[2] = ao2mo.general(eri_ab, \
                               (C[0], C[0], C[1], C[1]), compact=compact)
        eri_ab = None
        if restore:
            eri_new = np.zeros((3, norb, norb, norb, norb), dtype=np.double)
            eri_new[0] = ao2mo.restore(1, eri[0], norb)
            eri_new[1] = ao2mo.restore(1, eri[1], norb)
            eri_new[2] = ao2mo.restore(1, eri[2], norb)
            eri = eri_new
            eri_new = None

        return h1e, eri
