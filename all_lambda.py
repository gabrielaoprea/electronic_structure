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
from matplotlib import pyplot as plt

at_string ='Be 0 0 0'
mol = gto.M(
    atom = at_string,  # in Angstrom
    basis = '631g',
    #spin = 1,
)

# HF calculation 
myhf = scf.HF(mol)
myhf.kernel()

# HF results
fock_at = myhf.get_fock()
density = myhf.make_rdm1()
hcore = myhf.get_hcore()
mo_en = myhf.mo_energy
orb = myhf.mo_coeff
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))

# orbital swap for Delta-SCF if needed
#orb[:,[0, 5]] = orb[:,[5, 0]]

# creation of the Hilbert space + operators that are needed
perm = pt.create_permutations(mol, mo_en)
perm.reverse()
fock_mo = orb.transpose().dot(fock_at).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
fock, h = pt.get_full_matrices(nuc, hcore_mo, fock_mo, mo_integrals, perm)
h1 = h - fock

# calculation to get eigenvalues for all lambda
lam_range = np.arange(-2, 2, 0.1)
nfci = h.shape[0]
e =  np.empty((nfci, 0))
print(e.shape)
for lam in lam_range:
    h_lam = fock + lam*h1
    e_new =  np.array([np.linalg.eigvalsh(h_lam)])
    e_new = np.transpose(e_new)
    e = np.append(e, e_new, axis = 1)
print(e)
print(e.shape)
for i in range(nfci):
    plt.plot(lam_range, e[i,:], linestyle = 'dotted', color = 'k')
plt.xlabel("λ")
plt.ylabel("Energy")
plt.legend(loc='best')
plt.title('Be eigenvalues in 6-31G')
ax = plt.gca()
#ax.set_ylim([-2,4])
plt.show()