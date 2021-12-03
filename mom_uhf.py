#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
import uhf_perturbation as pt 

mol = gto.Mole()
mol.atom = '''H 0 0 0; H 0 0 1.5'''#in Angstrom
mol.basis = 'sto3g'
mol.charge = 0
mol.spin = 0
mol.build()
myhf = scf.UHF(mol)
myhf.scf()
orb = myhf.mo_coeff
occ = myhf.mo_occ
occ[1][0]=0
occ[1][1]=1

b = scf.UHF(mol)
dm_u = b.make_rdm1(orb, occ)
b =  scf.addons.mom_occ(b,orb,occ)
b.scf(dm_u)

fock=b.get_fock()
#print(fock)
fock_alpha = fock[0]
fock_beta = fock[1]
density=b.make_rdm1()
hcore=b.get_hcore()
#print(hcore)
mo_en = b.mo_energy
perm = pt.create_permutations(mol, mo_en)
perm.reverse()
orb = b.mo_coeff
orb_beta = orb[0]
orb_beta[:,[0, 1]] = orb_beta[:,[1, 0]]
orb_alpha = orb[1]
fock_mo_alpha = orb_alpha.transpose().dot(fock_alpha).dot(orb_alpha)
fock_mo_beta = orb_beta.transpose().dot(fock_beta).dot(orb_beta)
hcore_mo_alpha = orb_alpha.transpose().dot(hcore).dot(orb_alpha)
hcore_mo_beta = orb_beta.transpose().dot(hcore).dot(orb_beta)
full = pt.full_fock(fock_mo_alpha, fock_mo_beta, perm)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
nuc = mol.energy_nuc()
h = pt.get_full_h(hcore_mo_alpha, hcore_mo_beta, full, ao_reshaped, perm, nuc, orb_alpha, orb_beta)
#print(full_fock)
psi, e = pt.mppt(h, full, 100)
print(e)
#plot_e(e)