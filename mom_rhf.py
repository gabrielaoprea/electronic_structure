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
import rhf_perturbation as pt
import approximants as apx

at_string ='H 0 0 0; H 0 0 {bond_length:.2f}'
e_list = []
bond_lengths = np.arange(0.2,4.6,0.05)
for i in bond_lengths:
    mol = gto.M(
        atom = at_string.format(bond_length = i),  # in Angstrom
        basis = 'sto3g'
        #spin = 1,
    )
    myhf = scf.HF(mol)
    myhf.scf()

    orb = myhf.mo_coeff
    occ = myhf.mo_occ
    occ[0] = 0
    occ[1] = 2
    mo_en = myhf.mo_energy
    b = scf.HF(mol)
    #transformation = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4),0,0], [np.sin(np.pi/4), np.cos(np.pi/4),0,0],[0,0,1,0],[0,0,0,1]])
    #transformation = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
    #orb = orb.dot(transformation)
    dm_u = b.make_rdm1(orb, occ)
    b = scf.addons.mom_occ(b, orb, occ)
    b.scf(dm_u)
    perm = pt.create_permutations(mol)
    perm.reverse()
        
    fock=b.get_fock()
    density=b.make_rdm1()
    hcore=b.get_hcore()
    mo_en = b.mo_energy
    orb = b.mo_coeff

    for i in range(1,len(b.mo_occ)):
        if b.mo_occ[i] == 2:
            orb[:,[0, i]] = orb[:,[i, 0]]
    fock_mo = orb.transpose().dot(fock).dot(orb)
    hcore_mo = orb.transpose().dot(hcore).dot(orb)
    fock = pt.full_fock(fock_mo, perm)
    ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
    mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
    nuc = mol.energy_nuc()
    h = pt.get_full_h(hcore_mo, fock, mo_integrals, perm, nuc)
    print('Htot')
    print(h)
    psi, e = pt.mppt(h, fock, 50)

    np.set_printoptions(linewidth=10000,precision=16,suppress=True)
    e = list(e)
    r, p, q = apx.quadratic_approximant(e, 2)
    print(apx.get_values(r,p,q,1))
    discriminant = p**2 - 4*q*r
    print(discriminant.roots())

    lam = np.arange(0,5,0.2)
    l1 = []
    l2 = []
    el = []
    for i in lam:
        s = apx.get_values(r,p,q,i)
        l1.append(s[0])
        l2.append(s[1])
        el.append(sum(different_lambda(e, i)))
    plt.plot(lam,l1, color = 'b', label = 'Quadratic [2,2,2]')
    plt.plot(lam,l2, color = 'b')
    plt.plot(lam,el, linestyle = 'dashed', color = 'k', label ='Taylor')
    plt.xlabel("λ")
    plt.ylabel("E")
    plt.legend(loc='best')
    ax = plt.gca()
    ax.set_ylim([-5,2])
    plt.show()
    e_list.append(e)
print(e_list)