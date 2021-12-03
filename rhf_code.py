#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
import rhf_perturbation as pt

at_string ='H 0 0 0; H 0 0 {bond_length:.2f}'
e_list = []
bond_lengths = np.arange(0.2,4.6,0.05)
for i in bond_lengths:
    mol = gto.M(
        atom = at_string.format(bond_length = i),  # in Angstrom
        basis = 'sto3g',
        #spin = 1,
    )

    myhf = scf.HF(mol)
    myhf.kernel()

    fock = myhf.get_fock()
    density = myhf.make_rdm1()
    hcore = myhf.get_hcore()
    mo_en = myhf.mo_energy
    perm = pt.create_permutations(mol, mo_en)
    perm.reverse()
    orb = myhf.mo_coeff
    fock_mo = orb.transpose().dot(fock).dot(orb)
    hcore_mo = orb.transpose().dot(hcore).dot(orb)
    fock = pt.full_fock(fock_mo,perm)
    ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
    mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
    nuc = mol.energy_nuc()
    h = pt.get_full_h(hcore_mo, fock, mo_integrals, perm, nuc)
    psi, e = mppt(h, fock, 10)
    e = list(e)
    e_list.append(e)
print(e_list)