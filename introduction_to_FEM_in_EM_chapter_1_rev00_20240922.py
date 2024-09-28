#
# TITLE:
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: A.C.Polycarpou - Introduction to the Finite Element Method in Electromagnetics (2006)
#


import numpy as np
import sympy as sp


#==================================================
# 1.2 ELECTROSTATIC BVP AND THE ANALYTICAL SOLUTION
#

if False:

    # symbols
    x = sp.symbols('x')         # position
    ep0 = sp.symbols('ep0')     # vacuum permittivity
    epr = sp.symbols('epr')     # vacuum permittivity
    rho = sp.symbols('rho')     # charge density
    d = sp.symbols('d')         # distance between two metal plates
    v0 = sp.symbols('vo')       # electric potential at x = 0
    
    # functions
    v = sp.Function('v')        # electric potential

    # electromagnetism: electric displacement
    D = ( ep0 * epr ) * ( -v(x).diff(x, 1) )

    # poisson equation
    poisson_eq_left = D.diff(x, 1)
    poisson_eq_right = -rho
    poisson_eq = sp.Eq(poisson_eq_left, poisson_eq_right)
    
    # boundary conditions: Dirichlet or essential boundary conditions
    bc = {}
    bc[v(0)] = v0
    bc[v(d)] = 0.0
    
    # solve ODE w/ BCs
    analytic_solution = sp.dsolve(poisson_eq, v(x), ics=bc)
    analytic_solution = analytic_solution.rhs

    # simplify solution
    coefficients_dict = analytic_solution.as_coefficients_dict(x)
    for item in coefficients_dict.keys():
        print(item , ' > ', sp.simplify(coefficients_dict[item]))


#==================================================
# 1.3 THE FINITE ELEMENT METHOD
#
# numerical technique that is used to solve BVPs governed by a DE and a set of BCs
#
# representation of the domain w/ smaller subdomains called finite elements
#
# to form a linear system of equations, the governing DE and associated BCs must first be converted to
# an integro-differential formulation either
# (a) by minimizing a functional or
# (b) using weighted-residual method such as the Galerkin approach
#
# two methods that are used to obtain the FE equations
# (a) variational method and
# (b) weighted-residual method
#
# variational method requires construction of a functional which
# represents the energy associated with the BVP at hand
#
# functional is a function expressed in an integral form and has arguments that are functions themselves
# -> function of functions
# a stable or stationary solution to a BVP can be obtained by minimizing or maximizing the governing functional
# 
# weighted-residual method widely known as the Galerkin method
# forming a residual directly from the PDE that is associated with BVP under study
# does not require the use of a functional
# residual is formed by transferring all terms of the PDE on one side
# this residual is then multiplied bt as weight function and integrated over the domain of a single element
#
# major steps of Galerkin FEM for the solution of a BVP
# 1. discretize the domain using finite elements
# 2. choose proper interpolation functions (shape functions or basis functions)
# 3. obtain the corresponding linear equations for a single element by
#    first deriving the weak formulation of DE subject to a set of BCs
# 4. for the global matrix system of equations through the assembly of all elements
# 5. impose Dirichlet BCs 
# 6. solve the linear system of equations using linear algebra techniques
# 7. postprocess the results
#


#==================================================
# 1.4 DOMAIN DISCRETIZATION
#
# domain of the problem [0.0, d]
# Ne = number of elements
# Nn = number of nodes
# higher order elements
#


#==================================================
# 1.5 INTERPOLATION FUNCTIONS (shape functions)
#
#  element along x-axis   >   element along ξ-axis (master element)
#  regular coordinates    >   natural coordinate
#  [x1, x2]               >   [-1, +1]
#
#  V(ξ) = V1 * N1(ξ) + V2 * N2(ξ)
#
#  N1(ξ), N2(ξ) = interpolation functions
#
# number of interpolation functions per element =  number of nodes or DoF that belong to the element
#  DoF = degrees of freedom 
#
# N1(ξ) = ( 1 - ξ ) / 2
# N2(ξ) = ( 1 + ξ ) / 2
#

if False:
    x  = sp.symbols('x')
    xi = sp.symbols('xi')

    v1 = sp.symbols('v1')
    v2 = sp.symbols('v2')

    N1 = ( 1 - xi ) / 2
    N2 = ( 1 + xi ) / 2

    v = v1 * N1 + v2 * N2

    print(N1)
    print(N2)
    print(v)


#==================================================
# 1.6 THE METHOD OF WEIGHTED RESIDUAL: THE GALERKIN APPROACH
#

if False:
    
    # symbols
    x = sp.symbols('x')             # position
    x1, x2 = sp.symbols('x1 x2')    # position
    rho = sp.symbols('rho')         # charge density

    # functions
    v  = sp.Function('v')       # electric potential
    ep = sp.Function('ep')      # electric permittivity
    w  = sp.Function('w')       # weight function

    # electromagnetism: electric displacement
    D = ep(x) * ( -v(x).diff(x, 1) )

    # poisson equation
    poisson_eq_left = D.diff(x, 1)
    poisson_eq_right = rho

    # weighted residual
    wr = w(x) * (poisson_eq_left - poisson_eq_right)

    # weighted-integral equation
    wi = sp.Integral(wr, (x, x1, x2))
    
    print(wr)


#==================================================
# 1.8 IMPOSITION OF BOUNDARY CONDITIONS
#
# imposing boundary conditions (BCs)
# on the set of linear equations obtained from the weak formulation of the governing differential equation (DE)
#
# prior to imposing BCs,
# the global matrix system (GMS) is singular and, thus, cannot be solved to obtain a unique solution
#
# a nonsingluar matrix system is obtained after imposing BCs associated with a given BVP
#
# (a) dirichlet BCs: only the primary unknown variable
# (b) mixed BCs: both the promary unknown variable and its derivative
#     neumann BCs: special case of the mixed BCs
#
# weak formulation of the governing DE over the entire FE domain results in
# a system of N linear equations with N unknowns
#
# imposing M dirichlet BCs, the size of the final linear system of equations will be reduced to N - M
#
# K v = b    w/   imposing Vn = V0 (dirichlet BC)
# -> n-th row & n-th column of system matrix K must be eliminated
# -> n-th row of the unknown vector v must be eliminated
# -> n-th row of RHS vector b must be eliminated
# -> remaining RHS vector b must be updated
#    bi = bi - Kin V0  for  i = 1, 2, ..., N; i != n
# 
# method of elimination 
# the global node at which the Dirichlet BC is imposed must be eliminated
# reducing the size of the governing matrix system
# 
# programming method of elimination 
# more convenient to number the global nodes of the FE mesh in such a way that 
# the nodes which correspond to Dirichlet BC sappear last
#
# ep * dv / dx + alpha * v = beta
#
# global RHS vector: b = f + d
#     [ D(1)1   ]   [ -ep(1) * dvd/dx @x(1)1  ]
# d = | 0 ... 0 ] = | 0 ... 0.                |
#     [ -D(Ne)2 ]   [  ep(Ne) * dv/dx @x(Ne)2 ]
#
# ep(Ne) * dv/dx @x(Ne)2 = beta - alpha * v@N
#
# Transferring alpha * v@N to the LHS of the matrix system is
# equivalent to adding constant alpha to the KNN entry of the global coefficient matrix
#


#==================================================
# 1.9 FINITE ELEMENT SOLUTION OF THE ELECTROSTATIC BOUNDARY-VALUE PROBLEM
#
# computing the electric potential distribution between two parallel plates 

if True:

    # universal constants

    ep0 = 8.85e-12
    
    # user inputs

    d = 0.08        # distance between two plates [m]
    vl = 1.0        # electric potential at the leftmost plate [V]
    vr = 0.0        # electric potential at the rightmost plate [V]

    epr = 1.0       # dielectric constant in the region between the plates
    rho = 1e-8      # charge density in the region between the plates [C/m^3]

    # FEM input
    
    Ne = 10

    # length of each element

    le = d / Ne
    
    # calculating the element coefficient matrix Ke

    Ke = np.zeros((2, 2), dtype=float)

    Ke[0, 0] =  (ep0 * epr) / le
    Ke[0, 1] = -(ep0 * epr) / le
    Ke[1, 0] = -(ep0 * epr) / le
    Ke[1, 1] =  (ep0 * epr) / le

    # calculating fe

    fe = np.zeros((2, 1), dtype=float)

    fe[0] = - rho * le / 2.0
    fe[1] = - rho * le / 2.0

    # nodes

    Nn = Ne + 1

    # global matrix system

    K = np.zeros((Nn, Nn), dtype=float)
    f = np.zeros((Nn, 1), dtype=float)
    
    for cnt_e in range(Ne):
        K[cnt_e:cnt_e+2, cnt_e:cnt_e+2] += Ke
        f[cnt_e:cnt_e+2] += fe

    # imposing dirichlet BCs of vl & vr

    K_dbc = K[1:-1, 1:-1]
    
    f_dbc = f[1:-1]
    f_dbc[0] += - K[1,0] * vl           # left
    f_dbc[-1] += - K[-2,-1] * vr        # right

    # solve global matrix system

    v_dbc = np.linalg.solve(K_dbc, f_dbc)

    # post processing 1: element

    xe = []
    
    for cnt_n in range(Ne):
        if cnt_n == 0:
            xe.append(0.0)
        xe.append(xe[-1] + le)
        
    xe = np.hstack(xe)
    ve = np.vstack([np.array([vl]), v_dbc, np.array([vr])])
    
    # post processing 2: electric potential
    
    x = []
    v = []

    div = 100

    x_xi = np.linspace(-1.0, +1.0, div, dtype=float)

    for cnt_xe in range(len(xe)):
        if cnt_xe < len(xe)-1:
            if cnt_xe == 0:
                x = np.linspace(xe[cnt_xe], xe[cnt_xe+1], div)
                v = ve[cnt_xe] * (xe[cnt_xe+1] - x) / (xe[cnt_xe+1] - xe[cnt_xe]) + \
                    ve[cnt_xe+1] * (x - xe[cnt_xe]) / (xe[cnt_xe+1] - xe[cnt_xe])
                x = x[:-1]
                v = v[:-1]
            else:
                temp_x = np.linspace(xe[cnt_xe], xe[cnt_xe+1], div)
                temp_v = ve[cnt_xe] * (xe[cnt_xe+1] - temp_x) / (xe[cnt_xe+1] - xe[cnt_xe]) + \
                         ve[cnt_xe+1] * (temp_x - xe[cnt_xe]) / (xe[cnt_xe+1] - xe[cnt_xe])
                temp_x = temp_x[:-1]
                temp_v = temp_v[:-1]
                x = np.hstack([x, temp_x])
                v = np.hstack([v, temp_v])

    # post processing 3: electric field

    xe2 = (xe[1:] + xe[:-1]) / 2.0
    exe = -(ve[1:] - ve[:-1]) / (xe[1:] - xe[:-1])

    x2 = x[1:]
    ex = -(v[1:] - v[:-1]) / (x[1:] - x[:-1])
    
    # plot

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(xe, ve, 'o')
    ax[0].plot(x,v, ':')
    ax[0].grid(ls=':')
    ax[1].plot(xe2, exe, 'o')
    ax[1].plot(x2, ex, ':')
    ax[1].grid(ls=':')
    plt.show()






