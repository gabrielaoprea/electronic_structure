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
#bond_lengths = np.arange(0.2,4.6,0.05)
bond_lengths = [0.74]
approxim = []
for i in bond_lengths:
    mol = gto.M(
        atom = "Ne 0 0 0",  # in Angstrom
        basis = '631g'
        #spin = 1,
    )
    myhf = scf.HF(mol)
    myhf.scf()

    orb = myhf.mo_coeff
    occ = myhf.mo_occ
    occ[1] = 0
    occ[5] = 2
    mo_en = myhf.mo_energy
    b = scf.HF(mol)
    transformation = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4),0,0], [np.sin(np.pi/4), np.cos(np.pi/4),0,0],[0,0,1,0],[0,0,0,1]])
    #transformation = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
    #orb = orb.dot(transformation)
    dm_u = b.make_rdm1(orb, occ)
    b = scf.addons.mom_occ(b, orb, occ)
    b.scf(dm_u)
    
        
    fock=b.get_fock()
    density=b.make_rdm1()
    hcore=b.get_hcore()
    mo_en = b.mo_energy
    orb = b.mo_coeff
    perm = pt.create_permutations(mol, mo_en)
    perm.reverse()
    orb = b.mo_coeff
    occ = b.mo_occ
    l = 0
    for i in range(len(occ)):
        if occ[i] == 2:
            orb[:,[l, i]] = orb[:,[i, l]]
            l +=1
    fock_mo = orb.transpose().dot(fock).dot(orb)
    hcore_mo = orb.transpose().dot(hcore).dot(orb)
    ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
    mo_integrals = pt.get_mo_integrals(ao_reshaped, orb)
    nuc = mol.energy_nuc()
    fock, h = pt.get_full_matrices(hcore_mo, fock_mo, mo_integrals, perm, nuc)
    psi, e = pt.mppt(h, fock, 100)
    #r, p, q = apx.quadratic_approximant(e, 4)
    #approxim.append(apx.get_values(r,p,q,1)[0])
    np.set_printoptions(linewidth=10000,precision=16,suppress=True)
    print(e)
    print(np.cumsum(e))
#print(approxim)
