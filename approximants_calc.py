#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.interpolate import pade
from numpy.polynomial import polynomial
from matplotlib import pyplot as plt
import rhf_perturbation as rhfpt
import uhf_perturbation as uhfpt
import approximants as apx

mol = gto.M(
    atom = 'H 0 0 0; H 0 0 2.5',  # in Angstrom
    basis = 'sto3g',
    #spin = 1,
)

myhf = mol.RHF().run()

fock=myhf.get_fock()
density=myhf.make_rdm1()
hcore=myhf.get_hcore()
mo_en = myhf.mo_energy

perm = rhfpt.create_permutations(mol, mo_en)
perm.reverse()
orb = myhf.mo_coeff
#orb[:,[0, 2]] = orb[:,[2, 0]]
fock_mo = orb.transpose().dot(fock).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
full_fock = rhfpt.full_fock(fock_mo, perm)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = rhfpt.get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
h = rhfpt.get_full_h(hcore_mo, full_fock, mo_integrals, perm, nuc)
psi, e = rhfpt.mppt(h, full_fock, 20)
p,q = pade(e,2)
print(p(1)/q(1))
print(q)
r, p, q = apx,quadratic_approximant(e, 4)
print(apx.get_values(r,p,q,1))
discriminant = p**2 - 4*q*r
print(discriminant.roots())

lam = np.arange(0,5,0.2)
l1 = []
l2 = []
el = []
for i in lam:
    s = get_values(r,p,q,i)
    l1.append(s[0])
    l2.append(s[1])
    el.append(sum(different_lambda(e, i)))
plt.plot(lam,l1, color = 'b', label = 'Quadratic [4,4,4]')
plt.plot(lam,l2, color = 'b')
plt.plot(lam,el, linestyle = 'dashed', color = 'k', label ='Taylor')
plt.xlabel("λ")
plt.ylabel("E")
plt.legend(loc='best')
ax = plt.gca()
ax.set_ylim([-5,2])
plt.show()
myhf.MP2().run()
print(np.cumsum(e))