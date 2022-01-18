#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
import uhf_perturbation as pt 

mol = gto.Mole()
mol.atom = '''Be 0 0 0'''#in Angstrom
mol.basis = 'sto3g'
mol.charge = 0
mol.spin = 0
mol.build()
myhf = scf.UHF(mol)
myhf.scf()
orb = myhf.mo_coeff
occ = myhf.mo_occ
occ[1][1] = 0
occ[1][3] = 1
occ[0][0] = 0
occ[0][3] = 1
occ[0][1] = 0
occ[0][2] = 1
#occ[1][3]=1
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
perm = pt.create_permutations(mol, mo_en[0])
perm.reverse()
orb = b.mo_coeff
occ = b.mo_occ
occ_alpha = occ[0]
occ_beta = occ[1]
orb_beta = orb[1]
orb_alpha = orb[0]
l_alpha = 0
l_beta = 0
for i in range(len(occ_alpha)):
	if occ_alpha[i] == 1:
		orb_alpha[:,[l_alpha, i]] = orb_alpha[:,[i, l_alpha]]
		l_alpha +=1
	if occ_beta[i] == 1:
		orb_beta[:,[l_beta, i]] = orb_beta[:,[i, l_beta]]
		l_beta+=1
fock_mo_alpha = orb_alpha.transpose().dot(fock_alpha).dot(orb_alpha)
fock_mo_beta = orb_beta.transpose().dot(fock_beta).dot(orb_beta)
hcore_mo_alpha = orb_alpha.transpose().dot(hcore).dot(orb_alpha)
hcore_mo_beta = orb_beta.transpose().dot(hcore).dot(orb_beta)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
alphas, betas, alpha_beta = pt.get_mo_integrals(ao_reshaped, orb_alpha, orb_beta)
nuc = mol.energy_nuc()
full, h = pt.get_full_matrices(nuc, hcore_mo_alpha, hcore_mo_beta, fock_mo_alpha, fock_mo_beta,alphas, betas, alpha_beta, perm)
print(h)
psi, e = pt.degenerate_pt_mom(h, full, 100)
#print(e)
#print(np.cumsum(e))
#plot_e(e)