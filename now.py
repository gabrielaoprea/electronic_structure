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
np.set_printoptions(threshold=np.inf)

at_string ='Be'
e_list = []
#bond_lengths = np.arange(0.2,4.6,0.05)
bond_lengths = [0.74]
mol = gto.M(
    atom = 'Be',  # in Angstrom
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
orb[:,[1, 5]] = orb[:,[5, 1]]
print("ENPT for Be")
print("1s2 3s2:")
fock_mo = orb.transpose().dot(fock_at).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
#fock = pt.full_fock(fock_mo,perm) 
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
dyson = pt.get_dyson(fock_mo, mo_integrals, perm)
print("Dyson partitioning:")
psi, e = pt.mppt(h, dyson, 100,0)
print('Energy corrections:')
print(e)
print("ENn")
print(np.cumsum(e))
print('Moller-Plesset partitioning:')
psi, e = pt.mppt(h, fock, 100, 0)
print('Energy corrections:')
print(e)
print('MPn')
print(np.cumsum(e))
print('Exact solutions:')
print(np.linalg.eigvalsh(h))

fock_at = myhf.get_fock()
density = myhf.make_rdm1()
hcore = myhf.get_hcore()
mo_en = myhf.mo_energy
perm = pt.create_permutations(mol, mo_en)
perm.reverse()
spin = pt.get_spin(perm)
orb = myhf.mo_coeff
orb_og = myhf.mo_coeff
orb[:,[0, 5]] = orb[:,[5, 0]]
print("ENPT for Be")
print("2s2 3s2:")
fock_mo = orb.transpose().dot(fock_at).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
#fock = pt.full_fock(fock_mo,perm) 
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
dyson = pt.get_dyson(fock_mo, mo_integrals, perm)
print("Dyson partitioning:")
psi, e = pt.mppt(h, dyson, 100,0)
print('Energy corrections:')
print(e)
print("ENn")
print(np.cumsum(e))
print('Moller-Plesset partitioning:')
psi, e = pt.mppt(h, fock, 100, 0)
print('Energy corrections:')
print(e)
print('MPn')
print(np.cumsum(e))
print('Exact solutions:')
print(np.linalg.eigvalsh(h))