#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
import scipy
import uhf_perturbation as pt

at_string ='H 0 0 0; H 0 0 {bond_length:.2f}'
#bond_lengths = np.arange(0.2,4.6,0.05)
bond_lengths = [0.74]
l = len(bond_lengths)
#bond_lengths = [0.74]
for i in bond_lengths:
    mol = gto.Mole()
    mol.atom = at_string.format(bond_length = i) #in Angstrom
    mol.basis = '631g'
    mol.charge = 0
    mol.spin = 0
    mol.build()
    myhf = scf.UHF(mol)
    myhf.kernel()

    fock=myhf.get_fock()
    fock_alpha = fock[0]
    fock_beta = fock[1]
    density=myhf.make_rdm1()
    hcore=myhf.get_hcore()
    mo_en = myhf.mo_energy
    perm = pt.create_permutations(mol, mo_en[0])
    print(perm)
    perm.reverse()
    orb = myhf.mo_coeff
    orb_alpha = orb[0]
    #orb_alpha[:,[0, 1]] = orb_alpha[:,[1, 0]]
    orb_beta = orb[1]
    fock_mo_alpha = orb_alpha.transpose().dot(fock_alpha).dot(orb_alpha)
    fock_mo_beta = orb_beta.transpose().dot(fock_beta).dot(orb_beta)
    hcore_mo_alpha = orb_alpha.transpose().dot(hcore).dot(orb_alpha)
    hcore_mo_beta = orb_beta.transpose().dot(hcore).dot(orb_beta)
    ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
    alphas, betas, alpha_beta = pt.get_mo_integrals(ao_reshaped, orb_alpha, orb_beta)
    nuc = mol.energy_nuc()
    full, h = pt.get_full_matrices(nuc, hcore_mo_alpha, hcore_mo_beta, fock_mo_alpha, fock_mo_beta,alphas, betas, alpha_beta, perm)
    #h = pt.get_full_h(hcore_mo_alpha, hcore_mo_beta, full, ao_reshaped, perm, nuc, orb_alpha, orb_beta)
    psi, e = pt.degenerate_pt(h, full, 10)
