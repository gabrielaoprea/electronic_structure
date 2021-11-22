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
    '''
    Argument: 
        array which represents the difference between 2 states - i.e. array of 0, 1 and -1 (0 if no change, 1 if electrom moved ups, -1 if electron moved down)
    Returns:
        2 lists with the initial and final positions of electrons that change place.
    '''
    pos_i = []
    pos_f = []
    for i in range(len(state)):
        if state[i] == 1:
            pos_i.append(i)
        if state[i] == -1:
            pos_f.append(i)
    return pos_i, pos_f

def electron_pos(state):
    '''
    Argument:
        list of 1s and 0s representing the electrons
    Returns:
        list of position of electrons in the orbitals (i.e. no. of each orbital which has electrons).
    '''
    pos = []
    for i in range(len(state)):
        if state[i] == 1:
            pos.append(i)
    return pos

def difference(es_1, es_2):
    '''
    Arguments:
        2 states of the form defined in the creation of the permutations (list of alpha electrons, list of beta electrons)
    Returns:
        n_alpha, n_beta - no of electrons with spin alpha and beta that have changed position during the transition
        change_alpha, change_beta - positions in which thesel electrons where in the initial and final states (first half of the list = initial, second halfof the list = final)
    '''
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
    Returns a list of all pairs of permutations that differ by 1 or 2 electrons. Each element in the list has the form:
        [i, j, d_alpha, d_beta, change_alpha, change_beta]
        i, j - #of the permutations in the pair
        d_alpha, d_beta, change_alpha, change_beta - as defined in the function difference
    '''
    n = len(list_of_permutations)
    pairs = []
    for i in range(0, n):
        for j in range(0, n):
            d_alpha, d_beta, change_alpha, change_beta = difference(list_of_permutations[i],list_of_permutations[j])
            if d_alpha + d_beta <3:
                pairs.append([i, j, d_alpha, d_beta, change_alpha, change_beta])
    return pairs

def full_fock(mo_fock, perm):
    '''
    Arguments:
        mo_fock - Fock matrix in the molecularorbital basis
        perm - list of all possiblepermutations in the Hilbert space
    Returns:
        Fock matrix in the full Hilbert space
    '''
    n = len(perm)
    fock = np.zeros((n,n))
    for i in range(n):
        el = electron_pos(perm[i][0]) + electron_pos(perm[i][1])
        for j in el:
            fock[i,i] += mo_fock[j,j]
    pairs = get_pairs(perm)
    for i in pairs:
        if i[2] + i[3] == 1:
            if i[2] == 1:
                fock[i[0], i[1]] = mo_fock[i[4][0],i[4][1]]
            if i[3] == 1:
                fock[i[0], i[1]] = mo_fock[i[5][0],i[5][1]]
    return fock

def get_mo_integrals(ao_integrals, orb):
    '''
    Arguments:
        ao_integrals - 2-electron integrals, as returned by ao2mo in the atomic basis
        orb - orbital coefficients
    Returns:
        mo_integrals - 2-electron integrals in the molecularorbital basis
    '''
    dim = len(orb)
    temp = np.zeros((dim,dim,dim,dim))  
    temp2 = np.zeros((dim,dim,dim,dim))  
    temp3= np.zeros((dim,dim,dim,dim))  
    mo_integrals = np.zeros((dim,dim,dim,dim))
    for i in range(0,dim):  
        for m in range(0,dim):  
            temp[i,:,:,:] += orb[m,i]*ao_integrals[m,:,:,:]  
        for j in range(0,dim):  
            for n in range(0,dim):  
                temp2[i,j,:,:] += orb[n,j]*temp[i,n,:,:]  
            for k in range(0,dim):  
                for o in range(0,dim):  
                    temp3[i,j,k,:] += orb[o,k]*temp2[i,j,o,:]  
                for l in range(0,dim):  
                    for p in range(0,dim):  
                        mo_integrals[i,j,k,l] += orb[p,l]*temp3[i,j,k,p]
    return mo_integrals

def get_full_h(hcore, fock, mo_integrals, perm, en_nuc):
    '''
    Arguments:
        hcore, fock - respective matrices in the molecular orbital basis
        mo_integrals - 2-electron integrals in the molecular orbital basis
        perm - list of all state sin the Hilbert space
        en_nuc - nuclear repulsion
    Returns:
        hamiltonian - full Hamiltonian in the Hilbert space
    '''
    n = len(perm)
    #print(mo_integrals)
    hamiltonian = np.zeros((n,n))
    hcore_en  = np.diagonal(hcore)
    for k in range(n):
        hamiltonian[k,k] = energy(hcore_en, perm[k][0]) + energy(hcore_en, perm[k][1])+ en_nuc
        alpha_pos = electron_pos(perm[k][0])
        beta_pos = electron_pos(perm[k][1])
        s = 0 
        for i in alpha_pos:
            for j in alpha_pos:
                s+= mo_integrals[i,i,j,j] - mo_integrals[i,j,i,j]
            for j in beta_pos:
                s+= mo_integrals[i,i,j,j]
        for i in beta_pos:
            for j in beta_pos:
                s+= mo_integrals[i,i,j,j] - mo_integrals[i,j,i,j]
            for j in alpha_pos:
                s+= mo_integrals[i,i,j,j]
        hamiltonian[k,k]+= s/2
    pairs = get_pairs(perm)
    for i in pairs:
        if i[2] + i[3] == 1:
            hamiltonian[i[0], i[1]] = fock[i[0],i[1]]
        if i[2] == 1 and i[3] == 1:
            hamiltonian[i[0],i[1]] = mo_integrals[i[4][1],i[4][0],i[5][1],i[5][0]]
    return hamiltonian

def project(matrix):
    '''
    Projects matrix out of the reference state.
    '''
    n = len(matrix)
    projector = np.identity(n)
    projector = np.delete(projector, 0, 1)
    m = projector.transpose().dot(matrix).dot(projector)
    return m

def first_order(h_0, h_1):
    '''
    Calculates the first order corrections to the wavefunction and energy, as these cannot be calculated using the iterative formula.
    '''
    n = len(h_0)
    eig = np.identity(n)
    psi_1 = np.zeros((n, 1))
    e_0 = h_0[0, 0]
    for i in range(1, n):
        g = np.reshape(eig[:,i], (len(eig),1))
        #print(g)
        psi_1 += (h_1[0, i]/(e_0 - h_0[i,i]))*g
        #print(psi_1)
    e_1 = h_1[0,0]
    return psi_1, e_1

def mppt(h_tot, h_0, order):
    '''
    Performs the perturbation theory.
    Arguments:
        h_tot - full Hamiltonian in the Hilbert space
        h_0 - unperturbed Hamiltonian, usually the Fock matrix
        order - the numberof corrections that will be calculated
    Returns:
        psi - list of all the eigenstate corrections, starting with the unperturbed eigenstate
        e - list of all energy corrections, starting with 0th order.
    '''
    n = len(h_tot)
    h_1 = h_tot-h_0
    e_0 = h_0[0,0]
    print('H0:')
    print(h_0)
    print('H1:')
    print(h_1)
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
    '''
    Calculates series for different values of lambda i.e. multiplies every element with lambda^order.
    '''
    e =[]
    for i in range(len(corr_list)):
        e.append(corr_list[i]*(l**i))
    return e

def shanks_transformation(s_list):
    '''
    Performes a Shanks transformation on a list.
    '''
    n = len(s_list)
    s = 0 
    shank_list = []
    for i in range(1, n-1):
        t = (s_list[i+1]*s_list[i-1]-s_list[i]**2)/(s_list[i+1]-2*s_list[i]+s_list[i-1])
        shank_list.append(t)
    return shank_list

def convergence_radius(e_list):
    '''
    Given a list of energy corrections, plots Ek+1/Ek and the Shanks transformed series.
    '''
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
    '''
    Plots the energy corrections - without the zeroth order which is generally too large and wouldn't allow other features to be observed.
    '''
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

def get_c_matrix(f):
    n = len(f)
    c = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            c[i,i-j]=f[j]
    return c

def get_beg(n):
    i_mat = np.zeros((3*n+2, n+1))
    for i in range(n+1):
        i_mat[i,i] = 1
    return i_mat

def concat_3(c, c2, im, n):
    fin = np.zeros((3*n+2, 3*n+3))
    for i in range(3*n+2):
        for j in range(n+1):
            fin[i,j] = im[i,j]
            fin[i,j+n+1] = c[i,j]
            fin[i,j+2*n+2] = c2[i,j]
    return fin

def quadratic_approximant(series, n):
    l = len(series)
    if(3*n+2>l):
        print('Cannot compute quadratic approximant to this order.')
        return 0 
    c = get_c_matrix(series)
    c_squared = c.dot(c)
    c_n = c[:3*n+2, :n+1]
    c_squared_n = c_squared[:3*n+2, :n+1]
    i_mat = get_beg(n)
    mat = concat_3(c_n, c_squared_n, i_mat,n)
    coeff = np.copy(mat[:,2*n+2])
    mat_f = np.delete(mat, 2*n+2,1)
    coeff = np.array([coeff])
    coeff = -coeff.transpose()
    solution = np.linalg.solve(mat_f, coeff)
    r = solution[0:(n+1)]
    p = solution[n+1:(2*n+2)]
    q = solution[(2*n+2):(3*n+2)]
    r = r.transpose()
    r = r[0].tolist()
    p = p.transpose()
    p = p[0].tolist()
    q = q.transpose()
    q = q[0].tolist()
    q = [1] + q
    r_poly = polynomial.Polynomial(r)
    p_poly = polynomial.Polynomial(p)
    q_poly = polynomial.Polynomial(q)
    return r_poly, p_poly, q_poly

def get_beg_2(n):
    i_mat = np.zeros((2*n+1, n+1))
    for i in range(n+1):
        i_mat[i,i] = 1
    return i_mat

def concat_2(c, im, n):
    fin = np.zeros((2*n+1, 2*n+2))
    for i in range(2*n+1):
        for j in range(n+1):
            fin[i,j] = im[i,j]
            fin[i,j+n+1] = c[i,j]
    return fin

def linear_approximant(series, n):
    l = len(series)
    if(2*n+1>l):
        print('Cannot compute linear approximant to this order.')
        return 0 
    c = get_c_matrix(serie)
    c_n = c[:2*n+1,:n+1]
    i_mat = get_beg_2(n)
    mat = concat_2(c_n, i_mat, n)
    coeff = np.copy(mat[:,n+1])
    mat_f = np.delete(mat, n+1,1)
    coeff = np.array([coeff])
    coeff = -coeff.transpose
    solution = np.linalg.solve(mat_f, coeff)
    p = solution[0:(n+1)]
    q = solution[n+1:(2*n+1)]
    p = p.transpose()
    p = p[0].tolist()
    q = q.transpose()
    q = q[0].tolist()
    q = [1] + q
    p_poly = polynomial.Polynomial(p)
    q_poly = polynomial.Polynomial(q)
    return p_poly, q_poly


def get_values(r, p, q, value):
    plus = (-p(value) + np.sqrt(p(value)**2 - 4*q(value)*r(value)))/(2*q(value))
    minus = (-p(value) - np.sqrt(p(value)**2 - 4*q(value)*r(value)))/(2*q(value))
    return plus, minus

mol = gto.M(
    atom = 'H 0 0 0; H 0 0 1.437707',  # in Angstrom
    basis = 'sto3g',
    unit = 'Bohr'
    #spin = 1,
)

myhf = mol.RHF().run()
#myhf.kernel()

fock=myhf.get_fock()
#print(fock)
density=myhf.make_rdm1()
hcore=myhf.get_hcore()
#print(hcore)
mo_en = myhf.mo_energy

perm = create_permutations(mol)
perm.reverse()
orb = myhf.mo_coeff
#print(orb)
#orb[:,[0, 1]] = orb[:,[1, 0]]
fock_mo = orb.transpose().dot(fock).dot(orb)
hcore_mo = orb.transpose().dot(hcore).dot(orb)
full_fock =full_fock(fock_mo, perm)
ao_reshaped = np.reshape(mol.intor("int2e"),(mol.nao, mol.nao, mol.nao, mol.nao))
mo_integrals = get_mo_integrals(ao_reshaped, orb)
nuc = mol.energy_nuc()
h = get_full_h(hcore_mo, full_fock, mo_integrals, perm, nuc)
psi, e = mppt(h, full_fock, 20)
p,q = pade(e,2)
print(p(1)/q(1))
print(q)
r, p, q = quadratic_approximant(e, 2)
print(get_values(r,p,q,1))
discriminant = p**2 - 4*q*r
#print(discriminant)
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
#print(el)
#print(l1)
#print(l2)
plt.plot(lam,l1, color = 'b', label = 'Quadratic [2,2,2]')
plt.plot(lam,l2, color = 'b')
plt.plot(lam,el, linestyle = 'dashed', color = 'k', label ='Taylor')
plt.xlabel("lambda")
plt.ylabel("E")
plt.legend(loc='best')
ax = plt.gca()
ax.set_ylim([-8,0])
#plt.show()
myhf.MP2().run()
print(np.cumsum(e))