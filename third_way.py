#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
import rhf_perturbation as pt
import approximants as apx
from scipy.interpolate import pade

mol = gto.M(
    atom = 'H 0 0 0; H 0 0 0.74',  # in Angstrom
    basis = '631g',
    #spin = 1,
)

myhf = scf.HF(mol)
myhf.kernel()

fock_at = myhf.get_fock()
density = myhf.make_rdm1()
hcore = myhf.get_hcore()
mo_en = myhf.mo_energy
perm = pt.create_permutations(mol, mo_en)
perm.reverse()
spin = pt.get_spin(perm)
orb = myhf.mo_coeff
orb_og = myhf.mo_coeff
orb[:,[0, 1]] = orb[:,[1, 0]]
fock_mo = orb.transpose().dot(fock_at).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
print('Fock:')
print(fock)

density_2 = myhf.make_rdm1(orb)
myhf.scf(density_2)
fock=myhf.get_fock()
density=myhf.make_rdm1()
hcore=myhf.get_hcore()
mo_en = myhf.mo_energy
fock_mo = orb.transpose().dot(fock_at).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
print('Fock:')
print(fock)