#
# TITLE:
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: A.C.Polycarpou - Introduction to the Finite Element Method in Electromagnetics (2006)
#


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


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
# 
# numerical error between the FE solution (Vfe) and exact analytic solution (Vex)
# (1) the area bounded by two curves
#     as compared to the total area under the curve described by the exact solution
# (2) computing L2 norm which represents the distance betweem two solutions
#
# electric field obtained from the numerical approach is shown to be constant over element
# and discontinuous across element boundaries
#
# as the finite element mesh becomes increasingly denser, the numerical solution approaches
# the exact analytical solution
#

if False:

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


#==================================================
# 1.10 ONE-DIMENSIONAL HIGHER ORDER INTERPOLATION FUNCTIONS
#
# introducing higher order interpolation functions, quadratic and cubic
#
# numerical error will be substantially reduced with the use of
# higher order elements as opposed to linear elements
#
#
# (a) quadratic shape function used to interpolate the solution of BVP over an element
#
# two of these nodes conincide with the end of nodes of the element whereas
# the third one must be an interior node
#
# x1_e @node1, X3_e @node 3, x2_e @node2  ->  -1, 0, +1 @natural coordinate system
#
# ξ = 2 * (x - x3) / (x2 - x1)  where  x3 = (x1 + x2) / 2
# V(ξ) = V1_e * N1(ξ) + V2_e * N2(ξ) + V3_e * N3(ξ)
#
# N1(ξ), N2(ξ), N3(ξ): Lagrange shape function
#
# N1(ξ) = 1/2 * ξ * (ξ - 1)
# N2(ξ) = 1/2 * ξ * (ξ + 1)
# N3(ξ) = -1 * (ξ + 1) * (ξ - 1)
#
# x = x1_e * 1/2 * ξ * (ξ - 1) + x2_e * 1/2 * ξ * (ξ + 1) + x3_e * (1 + ξ) * (1 - ξ)
#   = x3_e + ( x2_e - x1_e ) / 2 * ξ
#
# ξ = 2 * ( x - x3_e ) / ( x2_e - x1_e )
#
# 
# (b) cubic elements, cubic shape functions
#
# two of these coincide with the end nodes of the element and other two correspond to interior points
#
# x1_e @node1, X3_e @node 3, X4_e @node 4, x2_e @node2  ->  -1, -1/3, +1/3, +1 @natural coordinate system 
#
# V(ξ) = V1_e * N1(ξ) + V2_e * N2(ξ) + V3_e * N3(ξ) + V4_e * N4(ξ)
# x(ξ) = x1_e * N1(ξ) + x2_e * N2(ξ) + x3_e * N3(ξ) + x4_e * N4(ξ)
#
# N1(ξ) = -9/16 * (ξ + 1/3) * (ξ - 1/3) * (ξ - 1)
# N2(ξ) = +9/16 * (ξ + 1) * (ξ + 1/3) * (ξ - 1/3)
# N3(ξ) = 27/16 * (ξ + 1) * (ξ - 1/3) * (ξ - 1)
# N4(ξ) = -27/16 * (ξ + 1) * (ξ + 1/3) * (ξ - 1)
#
# x3_e = (2 * x1_e + x2_e) / 3
# x4_e = (x1_e + 2 * x2_e) / 3
#
# ξ = 2 * (x - xc_e) / (x2_e - x1_e)  where  xc_e = (x1_e + x2_e) / 2
#
#



#==================================================
# 1.11 ELEMENT MATRIX AND RIGHT-HAND-SIDE VECTOR USING QUADRATIC ELEMENTS
#
# weak formulation
#
# entries of the element coefficient matrix
#
# Kij_e = int_(x1_e)^(x2_e) [dNi/dx] * ep_e * [dNj/dx] dx  for  i, j = 1, 2, 3
#
# entries of RHS vector
#
# fi_e = int_(x1_e)^(x2_e) Ni * rho_v dx  for  i = 1, 2, 3
#

if True:

    # symbols
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    le = sp.symbols('le')
    x = sp.symbols('x')
    xi = sp.symbols('xi')
    ep = sp.symbols('ep')
    rho = sp.symbols('rho')

    # functions 
    N1 = sp.Function('N1')
    N2 = sp.Function('N2')
    N3 = sp.Function('N3')
    
    # x coordinate, xi coordinate
    dxi_dx = sp.diff(sp.Rational(+2, 1) * (x-x3) / (x2-x1), x)
    dxi_dx = dxi_dx.subs(x2-x1, le)
    dx_dxi = 1/ dxi_dx

    # Lagrange shape function
    N1 = sp.Rational(+1, 2) * xi * (xi - 1)
    N2 = sp.Rational(+1, 2) * xi * (xi + 1)
    N3 = (1 + xi) * (1 - xi)
    
    dN1_dxi = sp.diff(N1, xi)
    dN2_dxi = sp.diff(N2, xi)
    dN3_dxi = sp.diff(N3, xi)

    # element coefficent matrix
    K11_e = sp.integrate( (dN1_dxi*dxi_dx) * ep * (dN1_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K12_e = sp.integrate( (dN1_dxi*dxi_dx) * ep * (dN2_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K13_e = sp.integrate( (dN1_dxi*dxi_dx) * ep * (dN3_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K21_e = sp.integrate( (dN2_dxi*dxi_dx) * ep * (dN1_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K22_e = sp.integrate( (dN2_dxi*dxi_dx) * ep * (dN2_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K23_e = sp.integrate( (dN2_dxi*dxi_dx) * ep * (dN3_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K31_e = sp.integrate( (dN3_dxi*dxi_dx) * ep * (dN1_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K32_e = sp.integrate( (dN3_dxi*dxi_dx) * ep * (dN2_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K33_e = sp.integrate( (dN3_dxi*dxi_dx) * ep * (dN3_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))

    Ke = sp.Matrix([[K11_e, K12_e, K13_e], [K21_e, K22_e, K23_e], [K31_e, K32_e, K33_e]])

    # RHS vector
    f1_e = sp.integrate( -rho * N1 * dx_dxi, (xi, -1, 1))
    f2_e = sp.integrate( -rho * N2 * dx_dxi, (xi, -1, 1))
    f3_e = sp.integrate( -rho * N3 * dx_dxi, (xi, -1, 1))

    fe = sp.Matrix([ [f1_e], [f2_e], [f3_e]])

    # print
    print('dxi/dx > ', dxi_dx)
    print('dx/dxi > ', dx_dxi)
    print(N1, ' > ', dN1_dxi)
    print(N2, ' > ', dN2_dxi)
    print(N3, ' > ', dN3_dxi)
    print('K11_e = ', K11_e)
    print('K12_e = ', K12_e)
    print('K13_e = ', K13_e)
    print('K21_e = ', K21_e)
    print('K22_e = ', K22_e)
    print('K23_e = ', K23_e)
    print('K31_e = ', K31_e)
    print('K32_e = ', K32_e)
    print('K33_e = ', K33_e)
    print('Ke = ', Ke)
    print('Ke.shape = ', Ke.shape)
    print('f1_e = ', f1_e)
    print('f2_e = ', f2_e)
    print('f3_e = ', f3_e)
    print('fe = ', fe)
    print('fe.shape = ', fe.shape)



#==================================================
# 1.12 ELEMENT MATRIX AND RIGHT-HAND-SIDE VECTOR USING CUBIC ELEMENTS
#
# weak formulation
#
# entries of the element coefficient matrix
#
# Kij_e = int_(x1_e)^(x2_e) [dNi/dx] * ep_e * [dNj/dx] dx  for  i, j = 1, 2, 3, 4
#
# entries of RHS vector
#
# fi_e = int_(x1_e)^(x2_e) Ni * rho_v dx  for  i = 1, 2, 3, 4
#

if False:

    # symbols
    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
    le = sp.symbols('le')
    x = sp.symbols('x')
    xi = sp.symbols('xi')
    ep = sp.symbols('ep')
    rho = sp.symbols('rho')

    # functions 
    N1 = sp.Function('N1')
    N2 = sp.Function('N2')
    N3 = sp.Function('N3')
    N4 = sp.Function('N4')
    
    # x coordinate, xi coordinate
    dxi_dx = sp.diff(sp.Rational(+2, 1) * (x-(x2+x1)/2) / (x2-x1), x)
    dxi_dx = dxi_dx.subs(x2-x1, le)
    dx_dxi = 1/ dxi_dx

    # Lagrange shape function
    N1 = sp.Rational(-9, 16) * (xi + sp.Rational(+1, 3)) * (xi - sp.Rational(+1, 3)) * (xi - 1)
    N2 = sp.Rational(+9, 16) * (xi + 1) * (xi + sp.Rational(+1, 3)) * (xi - sp.Rational(+1, 3))
    N3 = sp.Rational(+27, 16) * (xi + 1) * (xi - sp.Rational(+1, 3)) * (xi - 1)
    N4 = sp.Rational(-27, 16) * (xi + 1) * (xi + sp.Rational(+1, 3)) * (xi - 1)

    dN1_dxi = sp.diff(N1, xi)
    dN2_dxi = sp.diff(N2, xi)
    dN3_dxi = sp.diff(N3, xi)
    dN4_dxi = sp.diff(N4, xi)

    # element coefficent matrix
    K11_e = sp.integrate( (dN1_dxi*dxi_dx) * ep * (dN1_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K12_e = sp.integrate( (dN1_dxi*dxi_dx) * ep * (dN2_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K13_e = sp.integrate( (dN1_dxi*dxi_dx) * ep * (dN3_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K14_e = sp.integrate( (dN1_dxi*dxi_dx) * ep * (dN4_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K21_e = sp.integrate( (dN2_dxi*dxi_dx) * ep * (dN1_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K22_e = sp.integrate( (dN2_dxi*dxi_dx) * ep * (dN2_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K23_e = sp.integrate( (dN2_dxi*dxi_dx) * ep * (dN3_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K24_e = sp.integrate( (dN2_dxi*dxi_dx) * ep * (dN4_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K31_e = sp.integrate( (dN3_dxi*dxi_dx) * ep * (dN1_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K32_e = sp.integrate( (dN3_dxi*dxi_dx) * ep * (dN2_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K33_e = sp.integrate( (dN3_dxi*dxi_dx) * ep * (dN3_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K34_e = sp.integrate( (dN3_dxi*dxi_dx) * ep * (dN4_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K41_e = sp.integrate( (dN4_dxi*dxi_dx) * ep * (dN1_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K42_e = sp.integrate( (dN4_dxi*dxi_dx) * ep * (dN2_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K43_e = sp.integrate( (dN4_dxi*dxi_dx) * ep * (dN3_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))
    K44_e = sp.integrate( (dN4_dxi*dxi_dx) * ep * (dN4_dxi*dxi_dx) * dx_dxi, (xi, -1, +1))

    Ke = sp.Matrix([ [K11_e, K12_e, K13_e, K14_e], [K21_e, K22_e, K23_e, K24_e], [K31_e, K32_e, K33_e, K34_e], [K44_e, K42_e, K43_e, K44_e] ])

    # RHS vector
    f1_e = sp.integrate( -rho * N1 * dx_dxi, (xi, -1, 1))
    f2_e = sp.integrate( -rho * N2 * dx_dxi, (xi, -1, 1))
    f3_e = sp.integrate( -rho * N3 * dx_dxi, (xi, -1, 1))
    f4_e = sp.integrate( -rho * N4 * dx_dxi, (xi, -1, 1))

    fe = sp.Matrix([ [f1_e], [f2_e], [f3_e], [f4_e]])
    
    #
    print('dxi/dx > ', dxi_dx)
    print('dx/dxi > ', dx_dxi)
    print(N1, ' > ', dN1_dxi)
    print(N2, ' > ', dN2_dxi)
    print(N3, ' > ', dN3_dxi)
    print(N4, ' > ', dN4_dxi)
    print('K11_e = ', K11_e)
    print('K12_e = ', K12_e)
    print('K13_e = ', K13_e)
    print('K14_e = ', K14_e)
    print('K21_e = ', K21_e)
    print('K22_e = ', K22_e)
    print('K23_e = ', K23_e)
    print('K24_e = ', K24_e)
    print('K31_e = ', K31_e)
    print('K32_e = ', K32_e)
    print('K33_e = ', K33_e)
    print('K34_e = ', K34_e)
    print('K41_e = ', K41_e)
    print('K42_e = ', K42_e)
    print('K43_e = ', K43_e)
    print('K44_e = ', K44_e)
    print('Ke = ', Ke)
    print('Ke.shape = ', Ke.shape)
    print('f1_e = ', f1_e)
    print('f2_e = ', f2_e)
    print('f3_e = ', f3_e)
    print('f4_e = ', f4_e)
    print('fe = ', fe)
    print('fe.shape = ', fe.shape)



#==================================================
# 1.13 POSTPROCESSING OF THE SOLUTION: QUADRATIC ELEMENTS
#
# primary unknown quantity = electrostatic potential
# secondary unknown quantity = electric field
#
# [ V1_e V3_e V2_e ]
#
# V = V1_e * N1 + V2_e * N2 + V3_e * N3 
#   = V1_e * 1/2 * ξ * (ξ - 1) + V2_e * 1/2 * ξ * (ξ + 1) + V3_e * 1/2 * (1 + ξ) * (1 - ξ)
#
# Ex = - dV/dx = - dV/dξ * dξ/dx = - 2/le * dV/dξ
#    = - 2/le * ( V1_e * (ξ - 1/2) + V2_e * (ξ + 1/2) + V3_e * (-2 * ξ)  )
#
#


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
    
    Ne = 5

    # length of each element

    le = d / Ne
    
    # calculating the element coefficient matrix Ke

    Ke = np.zeros((3, 3), dtype=float)

    Ke[0, 0] =  +7.0 * (ep0 * epr) / (3.0 * le)
    Ke[0, 1] =  +1.0 * (ep0 * epr) / (3.0 * le)
    Ke[0, 2] =  -8.0 * (ep0 * epr) / (3.0 * le)
    Ke[1, 0] =  +1.0 * (ep0 * epr) / (3.0 * le)
    Ke[1, 1] =  +7.0 * (ep0 * epr) / (3.0 * le)
    Ke[1, 2] =  -8.0 * (ep0 * epr) / (3.0 * le)
    Ke[2, 0] =  -8.0 * (ep0 * epr) / (3.0 * le)
    Ke[2, 1] =  -8.0 * (ep0 * epr) / (3.0 * le)
    Ke[2, 2] = +16.0 * (ep0 * epr) / (3.0 * le)

    # calculating fe

    fe = np.zeros((3, 1), dtype=float)

    fe[0] = -1.0 * rho * le / 6.0
    fe[1] = -1.0 * rho * le / 6.0
    fe[2] = -2.0 * rho * le / 3.0

    # nodes

    Nn = 2 * Ne + 1

    # global matrix system

    K = np.zeros((Nn, Nn), dtype=float)
    f = np.zeros((Nn, 1), dtype=float)
    
    for cnt_e in range(Ne):
        # global node
        if cnt_e == 0:
            e_start = 2*cnt_e
        else:
            e_start = 2*cnt_e - 1    
        e_end = 2*cnt_e + 1
        e_quad = 2*cnt_e + 2

        g_node = [e_start, e_end, e_quad]

        # element coefficent matrix           
        for cnt_row in range(3):
            for cnt_col in range(3):
                K[g_node[cnt_row], g_node[cnt_col]] += Ke[cnt_row, cnt_col]
            f[g_node[cnt_row]] += fe[cnt_row]
        
    # imposing dirichlet BCs of vl & vr

    K_dbc = np.copy(K)
    f_dbc = np.copy(f)

    K_dbc[0, 0] = 1.0       # left
    K_dbc[-2,-2] = 1.0      # right

    f_dbc[0] = vl           # left
    f_dbc[-2] = vr          # right

    # solve global matrix system

    v_dbc = np.linalg.solve(K_dbc, f_dbc)

    # post processing 1: position to node number mapping

    pos_to_node = []

    for cnt_node in range(Nn):
        if cnt_node == 0:
            pos_to_node.append(cnt_node)
        elif cnt_node % 2 == 1:
            pos_to_node.append( 2 * ( cnt_node // 2 ) + 2 )
        elif cnt_node % 2 == 0:
            pos_to_node.append( 2 * ( cnt_node // 2 ) - 1 )

    print(pos_to_node)
    
    # post processing 2: element

    xe = []
    ve = []

    for cnt_pos in pos_to_node:
        if cnt_pos == 0:
            xe.append(0.0)
        else:
            xe.append(xe[-1] + le/2.0)

        ve.append(v_dbc[cnt_pos])

    ve = np.hstack(ve)

    xe = np.hstack(xe)
    ve = np.hstack(ve)

    # post processing 3: electric potential (interpolation)
    
    x = []
    v = []

    div = 100

    x_xi = np.linspace(-1.0, +1.0, div, dtype=float)

    for cnt_xe in range(len(xe)):
            
        # first element
        if cnt_xe == 0:
            start = cnt_xe
            quad = cnt_xe + 1
            end = cnt_xe + 2
        
            x = np.linspace(xe[start], xe[end], div)            
            v = ve[start] / 2 * x_xi * (x_xi - 1.0) + \
                ve[quad] * (1.0 + x_xi) * (1.0 - x_xi) + \
                ve[end] / 2 * x_xi * (x_xi + 1.0)

            x = x[:-1]
            v = v[:-1]

        # other element
        elif (cnt_xe % 2 == 0) and (cnt_xe < len(xe) - 1):
            start = cnt_xe
            quad = cnt_xe + 1
            end = cnt_xe + 2
        
            temp_x = np.linspace(xe[start], xe[end], div)
            temp_v = ve[start] / 2 * x_xi * (x_xi - 1.0) + \
                     ve[quad] * (1.0 + x_xi) * (1.0 - x_xi) + \
                     ve[end] / 2 * x_xi * (x_xi + 1.0)
            
            temp_x = temp_x[:-1]
            temp_v = temp_v[:-1]

            x = np.hstack([x, temp_x])
            v = np.hstack([v, temp_v])

    # post processing 4: electric field (interpolation)

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


#
# nonuniform charge distribution 
#
# rho_v = -rho_0 * ( 1 - x / d)**2
#
# -rho_0 at leftmost plate and a minimum value of 0 at the rightmost plate
#
# N1(ξ), N2(ξ), N3(ξ): Lagrange shape function
#
# N1(ξ) = 1/2 * ξ * (ξ - 1)
# N2(ξ) = 1/2 * ξ * (ξ + 1)
# N3(ξ) = -1 * (ξ + 1) * (ξ - 1)
#
# entries of RHS vector
#
# fi_e = int_(x1_e)^(x2_e) Ni * rho_v dx  for  i = 1, 2, 3
#
# 
# 
#
#
#    

if True:

    # symbols
    x = sp.symbols('x')         # position
    ep0 = sp.symbols('ep0')     # vacuum permittivity
    epr = sp.symbols('epr')     # vacuum permittivity
    rho0 = sp.symbols('rho0')     # charge density
    d = sp.symbols('d')         # distance between two metal plates
    v0 = sp.symbols('vo')       # electric potential at x = 0
    
    # functions
    v = sp.Function('v')        # electric potential

    # electromagnetism: electric displacement
    D = ( ep0 * epr ) * ( -v(x).diff(x, 1) )

    # poisson equation
    poisson_eq_left = D.diff(x, 1)
    poisson_eq_right = -rho0 * ( 1 - x/d )**2
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

    # shape functions

    N1 = 1/2

    # element RHS vector

    


















