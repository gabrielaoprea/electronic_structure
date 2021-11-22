#!/usr/bin/env python

from pyscf import gto, scf
import pyscf
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt

def energy(energy_list, permutation):
    '''
    Returns the energy of a specific electronic configuration. 
    Arguments:
        energy_list - list of energies of orbitals
        permutation - list of 0 and 1 representing occupancy of the orbitals with energies given by energy_list; 
                      NB: permutation is only for one spin - alpha or beta. 
    '''
    l = len(energy_list)
    s = 0
    for i in range(l):
        if permutation[i] == 1:
            s+=energy_list[i]
    return s

def create_permutations(molecule):
    '''
    Creates a list of all the possible permutations of alpha and beta electrons in the given molecule.
    The list is an N x 2 2D array - N = # of possible states; 2 = alpha electrons and beta electrons
    configurations are given separately.
    '''
    perm =[]
    n = len(mo_en) #number of orbitals 
    occ_alpha = (molecule.nelectron+molecule.spin)//2 #no of beta electrons
    occ_beta = molecule.nelectron-occ_alpha #no of alpha electrons
    alpha = [1]*occ_alpha+[0]*(n-occ_alpha)
    beta = [1]*occ_beta+[0]*(n-occ_beta)
    for i in multiset_permutations(alpha):
        for j in multiset_permutations(beta):
            perm.append([i,j])
    return perm

def get_es(energy_list, molecule, list_of_permutations):
    '''
    Returns the list of energies of all the possible electronic configurations of a molecule.
    Arguments:
        energy_list - list of energies either of orbitals or hcore;
        molecule - Mole object;
        list_of_permutations - all possible states expressed as permutations.
    '''
    f_energies = []
    for i in list_of_permutations:
        p_energy = energy(energy_list, i[0]) + energy(energy_list, i[1])
        f_energies.append(p_energy)
    return f_energies

def get_fock_e0(energy_list):
    '''
    Returns the list of energies with respect to the ground state.
    '''
    e0 = np.min(energy_list)
    f = []
    for i in range(len(energy_list)):
        f.append(energy_list[i]-e0)
    return f

def strip_of_0(list_of_values):
    return [value for value in list_of_values if value]

def make_diagonal_matrix(list_of_values):
    '''
    Returns a diagonal matrix - the values on the diagonal are given by the argument list_of_values.
    '''
    n = len(list_of_values)
    m = np.zeros((n,n))
    for i in range(n):
        m[i][i] = list_of_values[i]
    return m

def electron_change(state):
    pos_i = []
    pos_f = []
    for i in range(len(state)):
        if state[i] == 1:
            pos_i.append(i)
        if state[i] == -1:
            pos_f.append(i)
    return pos_i, pos_f

def electron_pos(state):
    pos = []
    for i in range(len(state)):
        if state[i] == 1:
            pos.append(i)
    return pos

def difference(es_1, es_2):
    es_1_alpha = np.array(es_1[0])
    es_1_beta = np.array(es_1[1])
    es_2_alpha = np.array(es_2[0])
    es_2_beta = np.array(es_2[1])
    d_alpha = es_1_alpha - es_2_alpha
    d_beta = es_1_beta - es_2_beta
    change_alpha_i, change_alpha_f = electron_change(d_alpha)
    change_beta_i,change_beta_f = electron_change(d_beta)
    n_alpha = len(change_alpha_i) 
    n_beta = len(change_beta_i)
    change_alpha = change_alpha_i + change_alpha_f 
    change_beta = change_beta_i + change_beta_f
    return n_alpha, n_beta, change_alpha, change_beta

def get_pairs(list_of_permutations):
    '''
    Returns a list of all the permutations that differ by 1 or 2 electrons. 
    '''
    n = len(list_of_permutations)
    pairs = []
    for i in range(0, n):
        for j in range(0, n):
            d_alpha, d_beta, change_alpha, change_beta = difference(list_of_permutations[i],list_of_permutations[j])
            if d_alpha + d_beta <3:
                pairs.append([i, j, d_alpha, d_beta, change_alpha, change_beta])
    return pairs

def full_fock(mo_fock_alpha, mo_fock_beta, perm):
    n = len(perm)
    fock = np.zeros((n,n))
    print(mo_fock_alpha)
    print(mo_fock_beta)
    for i in range(n):
        el_alpha = electron_pos(perm[i][0])
        el_beta = electron_pos(perm[i][1])
        for j in el_alpha:
            fock[i,i] += mo_fock_alpha[j,j]
        for j in el_beta:
            fock[i,i] += mo_fock_beta[j,j]
    print(fock)
    pairs = get_pairs(perm)
    for i in pairs:
        if i[2] + i[3] == 1:
            if i[2] == 1:
                fock[i[0], i[1]] = mo_fock_alpha[i[4][0],i[4][1]]
            if i[3] == 1:
                fock[i[0], i[1]] = mo_fock_beta[i[5][0],i[5][1]]
    return fock

def get_mo_integral_set(ao_integrals, orb_1, orb_2, orb_3, orb_4):
    dim = len(orb)
    temp = np.zeros((dim,dim,dim,dim))  
    temp2 = np.zeros((dim,dim,dim,dim))  
    temp3= np.zeros((dim,dim,dim,dim))  
    mo_integrals = np.zeros((dim,dim,dim,dim))
    for i in range(0,dim):  
        for m in range(0,dim):  
            temp[i,:,:,:] += orb_1[m,i]*ao_integrals[m,:,:,:]  
        for j in range(0,dim):  
            for n in range(0,dim):  
                temp2[i,j,:,:] += orb_2[n,j]*temp[i,n,:,:]  
            for k in range(0,dim):  
                for o in range(0,dim):  
                    temp3[i,j,k,:] += orb_3[o,k]*temp2[i,j,o,:]  
                for l in range(0,dim):  
                    for p in range(0,dim):  
                        mo_integrals[i,j,k,l] += orb_4[p,l]*temp3[i,j,k,p]
    return mo_integrals

def get_mo_integrals(ao_integrals, orb_alpha, orb_beta):
    alphas = get_mo_integral_set(ao_integrals,orb_alpha, orb_alpha, orb_alpha, orb_alpha)
    betas = get_mo_integral_set(ao_integrals, orb_beta, orb_beta, orb_beta, orb_beta)
    alpha_beta = get_mo_integral_set(ao_integrals, orb_alpha, orb_alpha, orb_beta, orb_beta)
    return alphas, betas, alpha_beta

def get_full_h(hcore_alpha, hcore_beta, fock, ao_integrals, perm, en_nuc, orb_aplha, orb_beta):
    n = len(perm)
    hamiltonian = np.zeros((n,n))
    hcore_en_alpha  = np.diagonal(hcore_alpha)
    hcore_en_beta = np.diagonal(hcore_beta)
    alphas, betas, alpha_beta = get_mo_integrals(ao_integrals, orb_alpha, orb_beta)
    #print(alphas)
    #print(betas)
    #print(alpha_beta)
    for k in range(n):
        hamiltonian[k,k] = energy(hcore_en_alpha, perm[k][0]) + energy(hcore_en_beta, perm[k][1])+ en_nuc
        alpha_pos = electron_pos(perm[k][0])
        beta_pos = electron_pos(perm[k][1])
        s = 0 
        for i in alpha_pos:
            for j in alpha_pos:
                s+= alphas[i,i,j,j] - alphas[i,j,i,j]
            for j in beta_pos:
                s+= alpha_beta[i,i,j,j]
        for i in beta_pos:
            for j in beta_pos:
                s+= betas[i,i,j,j] - betas[i,j,i,j]
            for j in alpha_pos:
                s+= alpha_beta[i,i,j,j]
        hamiltonian[k,k]+= s/2
    pairs = get_pairs(perm)
    for i in pairs:
        if i[2] + i[3] == 1:
            hamiltonian[i[0], i[1]] = fock[i[0],i[1]]          
        if i[2] == 1 and i[3] == 1:
            hamiltonian[i[0],i[1]] = alpha_beta[i[4][1],i[4][0],i[5][1],i[5][0]]
    return hamiltonian

def project(matrix):
    n = len(matrix)
    projector = np.identity(n)
    projector = np.delete(projector, 0, 1)
    m = projector.transpose().dot(matrix).dot(projector)
    return m

def first_order(h_0, h_1):
    n = len(h_0)
    eig = np.identity(n)
    psi_1 = np.zeros((n, 1))
    e_0 = h_0[0, 0]
    for i in range(1, n):
        g = np.reshape(eig[:,i], (len(eig),1))
        psi_1 += (h_1[0, i]/(e_0 - h_0[i,i]))*g
        #print(psi_1)
    e_1 = h_1[0,0]
    return psi_1, e_1

def mppt(h_tot, h_0, order):
    n = len(h_tot)
    h_1 = h_tot-h_0
    e_0 = h_0[0,0]
    psi_0 = np.zeros((n,1))
    psi_0[0,0] = 1
    psi_1, e_1 = first_order(h_0, h_1)
    e_2 = psi_0.transpose().dot(h_1).dot(psi_1)[0,0]
    psi = [psi_0, psi_1]
    psi_p = [np.delete(psi_0,0,0), np.delete(psi_1, 0, 0)]
    e = [e_0, e_1, e_2]
    h_tot_p = project(h_tot)
    h_0_p = project(h_0)
    h_1_p = project(h_1)
    inverse = inv(h_0_p - e_0*np.identity(n-1))
    for k in range(2, order):
        s = np.zeros((n-1,1))
        for r in  range(1, k):
            s += e[r]*psi_p[k-r]
        psi_new_p = -inverse.dot(s-h_1_p.dot(psi_p[k-1]))
        psi_new = np.vstack(([[0]], psi_new_p))
        psi_p.append(psi_new_p)
        psi.append(psi_new)
        e_new = psi_0.transpose().dot(h_1).dot(psi_new)[0,0]
        e.append(e_new)
    return psi, e

def different_lambda(corr_list, l):
    e =[]
    for i in range(len(corr_list)):
        e.append(corr_list[i]*(l**i))
    return e
def shanks_transformation(s_list):
    n = len(s_list)
    s = 0 
    shank_list = []
    for i in range(1, n-1):
        t = (s_list[i+1]*s_list[i-1]-s_list[i]**2)/(s_list[i+1]-2*s_list[i]+s_list[i-1])
        shank_list.append(t)
    return shank_list

def convergence_radius(e_list):
    ratio = []
    n = len(e_list)
    for i in range(n-1):
        ratio.append(abs(e_list[i+1]/e_list[i]))
    shanks = shanks_transformation(ratio)
    plt.plot(ratio,label = 'Initial series')
    plt.plot(shanks, label ='Shanks transformed series')
    plt.xlabel("k")
    plt.ylabel("E(k+1)/E(k)")
    plt.title("Determination of radius of convergence for H2 at bond length = 0.74 Angstrom")
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def plot_e(e):
    e_0 = e[0]
    x = []
    e.pop(0)
    for i in range(len(e)):
        x.append(i+1)
    plt.plot(x,e)
    plt.xlabel("Correction order")
    plt.ylabel("Energy correction")
    plt.title("Energy corrections for H2 bond length = 0.74 Angstrom")
    plt.show()

mol = gto.Mole()
mol.atom = '''H 0 0 0; H 0 0 1.570'''#in Angstrom
mol.basis = 'sto3g'
mol.charge = 0
mol.spin = 0
mol.build()
myhf = scf.UHF(mol)
myhf.kernel()

fock=myhf.get_fock()
#print(fock)
fock_alpha = fock[0]
fock_beta = fock[1]
#print(fock_alpha)
density=myhf.make_rdm1()
hcore=myhf.get_hcore()
#print(hcore)
mo_en = myhf.mo_energy
perm = create_permutations(mol)
perm.reverse()
orb = myhf.mo_coeff
orb_alpha = orb[0]
#orb_alpha[:,[0, 1]] = orb_alpha[:,[1, 0]]
orb_beta = orb[1]
fock_mo_alpha = orb_alpha.transpose().dot(fock_alpha).dot(orb_alpha)
fock_mo_beta = orb_beta.transpose().dot(fock_beta).dot(orb_beta)
hcore_mo_alpha = orb_alpha.transpose().dot(hcore).dot(orb_alpha)
hcore_mo_beta = orb_beta.transpose().dot(hcore).dot(orb_beta)
full_fock =full_fock(fock_mo_alpha, fock_mo_beta, perm)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
nuc = mol.energy_nuc()
h = get_full_h(hcore_mo_alpha, hcore_mo_beta, full_fock, ao_reshaped, perm, nuc, orb_alpha, orb_beta)
#print(h)
#print(full_fock)
psi, e = mppt(h, full_fock, 20)
print(e)
#plot_e(e)