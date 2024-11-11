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
# 2.1 INTRODUCTION
#
# nodal FEM applied to a generic 2D BVP in EM 
# ex) 2nd-order DE of a single dependent variable subjected to a set of BCs,
# Dirichlet type, Neumann type, or the mixed type
#
# finite elements in 2D: triangular or quadrilateral elements
#
# --------- major steps -----------
# a) discretization of 2D domain
# b) derivation of the weak formulation of the governing DE
# c) proper choice of interpolation functions
# d) derivation of the element matrices and vectors
# e) assembly of the global matrix system
# f) imposition of BCs
# g) solution of the global matrix system
# h) post-processing of the results
#

if False:

    # symbols
    x, y = sp.symbols('x, y')
    ep = sp.symbols('ep')
    rho = sp.symbols('rho')

    # functions
    v = sp.Function('v')
    
    # electromagnetism: electric displacement
    Dx = ep * -v(x, y).diff(x, 1)
    Dy = ep * -v(x, y).diff(y, 1)

    # 2D poisson's equation
    poisson_lhs  = Dx.diff(x, 1) + Dy.diff(y, 1)    # (2. 3)
    poisson_rhs  = rho                              # (2. 3)

    # display
    print(poisson_lhs)
    print(poisson_rhs)


#==================================================
# 2.2 DOMAIN DISTRETIZATION
#
# mesh generation
#
# 1) for a triangular mesh, the shape of triangles must be close to equilateral
# 2) for quadrilateral mesh, the shape of quadrilateral must be close to square
# 3) nodes must appear at source points
# 4) the FE mesh must accurately represent the geometrical domain of the problem
# 5) in regions where the solution is expected to have large variations,
#    the elements must be sufficiently small
# 6) avoid elements with very large aspect ration,
#    i.e., the ratio of the largest side to the smallest side
# 7) number the nodes in ascending order starting from 1.
#    the numbering of the nodes directly affects the bandwidth of the global matrix
# 8) there must be no overlap of elements
# 9) neighboring elements must share a common edge
# 10) an interior node (nonboundary node) must belong to at least three elements
#


#==================================================
# 2.3 INTERPOLATION FUNCTIONS
#
# proper interpolation functions
#
# 1) guarantee continuity of the primary unknown quantitiy
#    across interelement boundaries
# 2) at least once differentiable, must be complete polynomials
#    Lagrange polynomials

#-----
# 2.3.1 linear triangular element
#
# the nodes are locally numbered in a CCW direction (Jacobian definition)
#
# x y plane -> xi(ξ) eta(η) plane
#

if False:

    # symbols
    x, y = sp.symbols('x, y')                   # position
    xi, eta = sp.symbols('ξ, η')                # natural coordinates
    c1, c2, c3 = sp.symbols('c1, c2, c3')
    
    u1e, u2e, u3e = sp.symbols('u1e, u2e, u3e')
    x1e, x2e, x3e = sp.symbols('x1e, x2e, x3e')
    y1e, y2e, y3e = sp.symbols('y1e, y2e, y3e')

    x21e, x31e = sp.symbols('x21e, x31e')
    y21e, y31e = sp.symbols('y21e, y31e')

    #--------------------------------------------------------
    # linear representation of shape function N1: (2.7)
    N1_xieta = c1 + c2 * xi + c3 * eta

    # substitution: (2.8)
    N1_xieta_node1 = N1_xieta.subs({xi:0.0, eta:0.0}) - 1.0         # (0, 0)
    N1_xieta_node2 = N1_xieta.subs({xi:1.0, eta:0.0})               # (1, 0)
    N1_xieta_node3 = N1_xieta.subs({xi:0.0, eta:1.0})               # (0, 1)

    # solving linear equations
    N1_xieta_eqs = [N1_xieta_node1, N1_xieta_node2, N1_xieta_node3]
    N1_xieta_sol = sp.solve(N1_xieta_eqs, [c1, c2, c3], dict=True)

    # solution (2.9)
    for item in N1_xieta_sol[0].keys():
        N1_xieta = N1_xieta.subs(item, N1_xieta_sol[0][item])
    print(' (2.9)   N1 = ', N1_xieta)

    #--------------------------------------------------------
    # linear representation of shape function N2: (2.10)
    N2_xieta = c1 + c2 * xi + c3 * eta

    # substitution: (2.11)
    N2_xieta_node1 = N2_xieta.subs({xi:0.0, eta:0.0})           # (0, 0)
    N2_xieta_node2 = N2_xieta.subs({xi:1.0, eta:0.0}) - 1.0     # (1, 0)
    N2_xieta_node3 = N2_xieta.subs({xi:0.0, eta:1.0})           # (0, 1)

    N2_xieta_eqs = [N2_xieta_node1, N2_xieta_node2, N2_xieta_node3]
    N2_xieta_sol = sp.solve(N2_xieta_eqs, [c1, c2, c3], dict=True)

    # solution (2.12)
    for item in N2_xieta_sol[0].keys():
        N2_xieta = N2_xieta.subs(item, N2_xieta_sol[0][item])
    print(' (2.12)  N2 = ', N2_xieta)

    #--------------------------------------------------------
    # linear representation of shape function N3: (2.13)
    N3_xieta = c1 + c2 * xi + c3 * eta

    # substitution: (2.14)
    N3_xieta_node1 = N3_xieta.subs({xi:0.0, eta:0.0})           # (0, 0)
    N3_xieta_node2 = N3_xieta.subs({xi:1.0, eta:0.0})           # (1, 0)
    N3_xieta_node3 = N3_xieta.subs({xi:0.0, eta:1.0}) - 1.0     # (0, 1)

    N3_xieta_eqs = [N3_xieta_node1, N3_xieta_node2, N3_xieta_node3]
    N3_xieta_sol = sp.solve(N3_xieta_eqs, [c1, c2, c3], dict=True)

    # solution (2.15)
    for item in N3_xieta_sol[0].keys():
        N3_xieta = N3_xieta.subs(item, N3_xieta_sol[0][item])
    print(' (2.15)  N3 = ', N3_xieta)

    #--------------------------------------------------------
    # interpolation of x, y coordinates: (2.21)

    x_e = x1e * N1_xieta + x2e * N2_xieta + x3e * N3_xieta
    x_e = x_e.expand()
    x_e = x_e.collect(xi)
    x_e = x_e.subs(1.0*x2e-1.0*x1e, x21e)
    x_e = x_e.collect(eta)
    x_e = x_e.subs(1.0*x3e-1.0*x1e, x31e)
    
    y_e = y1e * N1_xieta + y2e * N2_xieta + y3e * N3_xieta
    y_e = y_e.expand()
    y_e = y_e.collect(xi)
    y_e = y_e.subs(1.0*y2e-1.0*y1e, y21e)
    y_e = y_e.collect(eta)
    y_e = y_e.subs(1.0*y3e-1.0*y1e, y31e)

    print(' (2.22)  x = ', x_e)
    print(' (2.22)  y = ', y_e)
    print('')


#-----
# 2.3.2 bilinear quadrilateral element
#

if True:

    # symbols
    x, y = sp.symbols('x, y')                           #
    xi, eta = sp.symbols('ξ, η')                        # natural coordinate system
    c1, c2, c3, c4 = sp.symbols('c1, c2, c3, c4')       #

    # rational
    pos_m1 = sp.Rational(-1.0, +1.0)
    pos_p1 = sp.Rational(+1.0, +1.0)
    val_p1 = sp.Rational(+1.0, +1.0)
    
    #--------------------------------------------------------
    # representation of bilinear quadrilateral shape function N1: (2.24)
    b_N1_xieta = c1 + c2 * xi + c3 * eta + c4 * xi * eta

    # substitution (2.25) (2.26)
    b_N1_xieta_node1 = b_N1_xieta.subs({xi:pos_m1, eta:pos_m1}) - val_p1
    b_N1_xieta_node2 = b_N1_xieta.subs({xi:pos_p1, eta:pos_m1})
    b_N1_xieta_node3 = b_N1_xieta.subs({xi:pos_p1, eta:pos_p1})
    b_N1_xieta_node4 = b_N1_xieta.subs({xi:pos_m1, eta:pos_p1})

    b_N1_xieta_eqs = [b_N1_xieta_node1, b_N1_xieta_node2, b_N1_xieta_node3, b_N1_xieta_node4]
    b_N1_xieta_sol = sp.solve(b_N1_xieta_eqs, [c1, c2, c3, c4], dict=True)

    # solution (2.27) (2.28)
    for item in b_N1_xieta_sol[0].keys():
        b_N1_xieta = b_N1_xieta.subs(item, b_N1_xieta_sol[0][item])

    # solution (2.29) (2.30)
    b_N1_xieta = b_N1_xieta.factor()
    print(' (2.29)  b_N1_xieta = ', b_N1_xieta)

    #--------------------------------------------------------
    # representation of bilinear quadrilateral shape function N2
    b_N2_xieta = c1 + c2 * xi + c3 * eta + c4 * xi * eta

    # substitution
    b_N2_xieta_node1 = b_N2_xieta.subs({xi:pos_m1, eta:pos_m1})
    b_N2_xieta_node2 = b_N2_xieta.subs({xi:pos_p1, eta:pos_m1}) - val_p1
    b_N2_xieta_node3 = b_N2_xieta.subs({xi:pos_p1, eta:pos_p1})
    b_N2_xieta_node4 = b_N2_xieta.subs({xi:pos_m1, eta:pos_p1})

    b_N2_xieta_eqs = [b_N2_xieta_node1, b_N2_xieta_node2, b_N2_xieta_node3, b_N2_xieta_node4]
    b_N2_xieta_sol = sp.solve(b_N2_xieta_eqs, [c1, c2, c3, c4], dict=True)

    # solution 
    for item in b_N2_xieta_sol[0].keys():
        b_N2_xieta = b_N2_xieta.subs(item, b_N2_xieta_sol[0][item])

    # solution (2.30)
    b_N2_xieta = b_N2_xieta.factor()
    print(' (2.30)  b_N2_xieta = ', b_N2_xieta)

    #--------------------------------------------------------
    # representation of bilinear quadrilateral shape function N3
    b_N3_xieta = c1 + c2 * xi + c3 * eta + c4 * xi * eta

    # substitution
    b_N3_xieta_node1 = b_N3_xieta.subs({xi:pos_m1, eta:pos_m1})
    b_N3_xieta_node2 = b_N3_xieta.subs({xi:pos_p1, eta:pos_m1})
    b_N3_xieta_node3 = b_N3_xieta.subs({xi:pos_p1, eta:pos_p1}) - val_p1
    b_N3_xieta_node4 = b_N3_xieta.subs({xi:pos_m1, eta:pos_p1})

    # solution
    b_N3_xieta_eqs = [b_N3_xieta_node1, b_N3_xieta_node2, b_N3_xieta_node3, b_N3_xieta_node4]
    b_N3_xieta_sol = sp.solve(b_N3_xieta_eqs, [c1, c2, c3, c4], dict=True)

    for item in b_N3_xieta_sol[0].keys():
        b_N3_xieta = b_N3_xieta.subs(item, b_N3_xieta_sol[0][item])

    # solution (2.30)
    b_N3_xieta = b_N3_xieta.factor()
    print(' (2.30)  b_N3_xieta = ', b_N3_xieta)

    #--------------------------------------------------------
    # representation of bilinear quadrilateral shape function N4
    b_N4_xieta = c1 + c2 * xi + c3 * eta + c4 * xi * eta
    
    b_N4_xieta_node1 = b_N4_xieta.subs({xi:pos_m1, eta:pos_m1})
    b_N4_xieta_node2 = b_N4_xieta.subs({xi:pos_p1, eta:pos_m1})
    b_N4_xieta_node3 = b_N4_xieta.subs({xi:pos_p1, eta:pos_p1})
    b_N4_xieta_node4 = b_N4_xieta.subs({xi:pos_m1, eta:pos_p1}) - val_p1

    N4_xieta_eqs = [N4_xieta_node1, N4_xieta_node2, N4_xieta_node3, N4_xieta_node4]
    N4_xieta_sol = sp.solve(N4_xieta_eqs, [c1, c2, c3, c4], dict=True)

    for item in N4_xieta_sol[0].keys():
        N4_xieta = N4_xieta.subs(item, N4_xieta_sol[0][item])

    N4_xieta = N4_xieta.factor()

    # display
    print('N1_xieta = ', N1_xieta)
    print('N2_xieta = ', N2_xieta)
    print('N3_xieta = ', N3_xieta)
    print('N4_xieta = ', N4_xieta)
    print('')


#
# arbitrary point (ξ, η) inside the master quadrilateral element
#

if False:

    # symbols
    x1_e, x2_e, x3_e, x4_e = sp.symbols('x1_e, x2_e, x3_e, x4_e')
    y1_e, y2_e, y3_e, y4_e = sp.symbols('y1_e, y2_e, y3_e, y4_e')
    
    # primary unknown quantity
    x = x1_e *  N1_xieta + x2_e *  N2_xieta + x3_e *  N3_xieta + x4_e *  N4_xieta
    y = y1_e *  N1_xieta + y2_e *  N2_xieta + y3_e *  N3_xieta + y4_e *  N4_xieta

    x = x.expand()
    y = y.expand()

    #
    x_xieta = x.coeff(xi*eta)
    x_temp = x - x_xieta * xi * eta
    x_temp = x_temp.simplify()
    x_xi = x_temp.coeff(xi)
    x_eta = x_temp.coeff(eta)
    x_const = x_temp - x_xi * xi - x_eta * eta
    x_const = x_const.simplify()

    #
    y_xieta = y.coeff(xi*eta)
    y_temp = y - y_xieta * xi * eta
    y_temp = y_temp.simplify()
    y_xi = y_temp.coeff(xi)
    y_eta = y_temp.coeff(eta)
    y_const = y_temp - y_xi * xi - y_eta * eta
    y_const = y_const.simplify()
    
    #
    print('x_const = ', x_const)
    print('x_xi coeff = ', x_xi)
    print('x_eta coeff = ', x_eta)
    print('x_xieta coeff = ', x_xieta)
    print('')
    print('y_const = ', y_const)
    print('y_xi coeff = ', y_xi)
    print('y_eta coeff = ', y_eta)
    print('y_xieta coeff = ', y_xieta)
    print('')

    
#==================================================
# 2.4 METHOD OF WEIGHTED RESIDUAL: THE GALERKIN APPROACH
#
# weak formulation of the problem
#   -> constructing the weighted residual for a single element with domain
#
# minimizing element residual
#   -> multiplying element residual with a weight function
#      then integrate the result over the area of the element,
#      and finally set the integral to zero
#

if False:

    # symbols
    x, y = sp.symbols('x, y')
    ax, ay, b, g = sp.symbols('ax, ay, b, g')

    # function
    u = sp.Function('u')
    w = sp.Function('w')
    
    # problem
    problem_lhs = ( ax * u(x, y).diff(x, 1) ).diff(x, 1) + ( ay * u(x, y).diff(y, 1) ).diff(y, 1) + b * u(x, y)
    problem_rhs = g

    # element residual
    element_residual = problem_lhs - problem_rhs
    
    element_residual_w = element_residual * w(x, y)
    element_residual_w = element_residual_w.expand()

    # 
    identity_x_lhs  = w(x, y) * ( ax * u(x, y).diff(x, 1) ).diff(x, 1)
    identity_x_rhs  = ( w(x, y) * ax * u(x, y).diff(x, 1) ).diff(x, 1)
    identity_x_rhs -= w(x, y).diff(x, 1) * ( ax * u(x, y).diff(x, 1) )

    # display
    print('problem LHS = ', problem_lhs)
    print('problem RHS = ', problem_rhs)
    print('element residual = ', element_residual)
    print('element residual_w = ', element_residual_w)
    print('identity_x_lhs = ', identity_x_lhs)
    print('identity_x_rhs = ', identity_x_rhs)
    

    
#==================================================
# 2.5 EVALUATION OF ELEMENT MATRICES AND VECTORS
#
# linear triangluar element
# -------------------------
#
# --- matrix M_e ---
# M_ij_e = - int_omega_e [ ax (∂Ni/∂x) (∂Nj/∂x) + ay (∂Ni/∂y) (∂Nj/∂y) ] dx dy  where ax and ay are constants 
#
# --- interpolation functions ---
# N1 = 1 - ξ - η
# N2 = ξ
# N3 = η
#
# --- x and y coordinates of any point inside an element
# x = x1_e + (x2_e - x1_e) ξ + (x3_e - x1_e) η
# y = y1_e + (y2_e - y1_e) ξ + (y3_e - y1_e) η
#
# --- differentiation
# ∂N/∂ξ = ∂N/∂x ∂x/∂ξ + ∂N/∂y ∂y/∂ξ
# ∂N/∂η = ∂N/∂x ∂x/∂η + ∂N/∂y ∂y/∂η
#
# [ ∂N/∂ξ ]     [ ∂x/∂ξ  ∂y/∂ξ  ] [ ∂N/∂x ]
# [         ]  =  [                   ] [         ]
# [ ∂N/∂η ]     [ ∂x/∂η  ∂y/∂η ] [ ∂N/∂y ]
#
#     [ ∂x/∂ξ  ∂y/∂ξ  ]    [ (x2_e - x1_e)  (y2_e - y1_e)  ]
# J = [                   ] =  [                               ]     Jacobian matrix
#     [ ∂x/∂η  ∂y/∂η ]     [ (x3_e - x1_e)  (y3_e - y1_e)  ]
#
# [ ∂N/∂x ]         [ ∂N/∂ξ ]
# [         ] = J^-1  [         ]
# [ ∂N/∂y ]         [ ∂N/∂η ] 
#
#
#        [ (y3_e - y1_e)  -(y2_e - y1_e)  ]
# J^-1 = [                                ]  /  det(J)
#        [ -(x3_e - x1_e)  (x2_e - x1_e)  ]
#
# where det(J) = (x2_e - x1_e) (y3_e - y1_e) - (x3_e - x1_e) (y2_e - y1_e) = 2 A_e
# where A_e = area of the triangle
#
# coordinate transformation
# -------------------------
#
# [ ∂N1/∂x ]          [ ∂N1/∂ξ ]    [ (y2_e - y3_e) ]
# [          ] = J^-1 * [          ]  = [               ]  /  det(J) 
# [ ∂N1/∂y ]          [ ∂N1/∂η ]    [ (x3_e - x1_e) ]
#
# [ ∂N2/∂x ]          [ ∂N2/∂ξ ]    [ (y3_e - y1_e) ]
# [          ] = J^-1 * [          ]  = [               ]  /  det(J) 
# [ ∂N2/∂y ]          [ ∂N2/∂η ]    [ (x1_e - x3_e) ]
#
# [ ∂N3/∂x ]          [ ∂N3/∂ξ ]    [ (y1_e - y2_e) ]
# [          ] = J^-1 * [          ]  = [               ]  /  det(J) 
# [ ∂N3/∂y ]          [ ∂N3/∂η ]    [ (x2_e - x1_e) ]
#
# to evaluate the double integral of M_ij_e,
# it is necessary to change the variables of integration from x and y to xi and eta.
# -> instead of interating over the triangular element on the regular coordinate system,
#    more convenient on the master trianlge using the natural coordinate system.
#
# M_ij_e = - int_omega_e [ ax (∂Ni/∂x) (∂Nj/∂x) + ay (∂Ni/∂y) (∂Nj/∂y) ] dx dy
#        = - int_omega_[0 1]_[0 1-eta]  f( x(ξ, η), y(ξ, η) )  |J|  dξ dη
#            (Jacobi transformation by German mathematician Carl Gustav Jacob Jacobi)   
#
#
#
#
#
#
#

if True:

    # symbols
    x1_e, x2_e, x3_e = sp.symbols('x1_e, x2_e, x3_e')
    y1_e, y2_e, y3_e = sp.symbols('y1_e, y2_e, y3_e')

    x12_e, x23_e, x31_e = sp.symbols('x12_e, x23_e, x31_e')
    y12_e, y23_e, y31_e = sp.symbols('y12_e, y23_e, y31_e')

    ax, ay = sp.symbols('ax, ay')

    xi, eta = sp.symbols('xi, eta')

    # interpolation functions
    N1 = 1 - xi - eta
    N2 = xi
    N3 = eta

    # x, y coordinates of any point inside an element
    x = N1 * x1_e + N2 * x2_e + N3 * x3_e
    x = x.expand()
    x = x.subs(x1_e - x2_e, x12_e)
    print(x.free_symbols)
    y = N1 * y1_e + N2 * y2_e + N3 * y3_e
    y = y.expand()
    
    x_xi = x.coeff(xi)
    x_eta = x.coeff(eta)
    x_const = x - x_xi * xi - x_eta * eta
    x_const = x_const.expand()

    y_xi = y.coeff(xi)
    y_eta = y.coeff(eta)
    y_const = y - y_xi * xi - y_eta * eta
    y_const = y_const.expand()

    # Jacobian matrix
    J = sp.Matrix([ [ x.diff(xi, 1), y.diff(xi, 1) ], [ x.diff(eta, 1), y.diff(eta, 1) ] ])
    J = J.expand()
    Jinv = J.inv()
    det_J = J.det()
    Jinv = Jinv * det_J     # normalization

    # coordinate transformation
    dN1_x_y = Jinv * sp.Matrix(2, 1, [N1.diff(xi, 1), N1.diff(eta, 1)])
    dN1_x_y = dN1_x_y       # / det_J is needed
    dN1_dx = dN1_x_y[0]     # / det_J is needed
    dN1_dy = dN1_x_y[1]     # / det_J is needed

    dN2_x_y = Jinv * sp.Matrix(2, 1, [N2.diff(xi, 1), N2.diff(eta, 1)])
    dN2_x_y = dN2_x_y       # / det_J is needed
    dN2_dx = dN2_x_y[0]     # / det_J is needed
    dN2_dy = dN2_x_y[1]     # / det_J is needed

    dN3_x_y = Jinv * sp.Matrix(2, 1, [N3.diff(xi, 1), N3.diff(eta, 1)])
    dN3_x_y = dN3_x_y       # / det_J is needed
    dN3_dx = dN3_x_y[0]     # / det_J is needed
    dN3_dy = dN3_x_y[1]     # / det_J is needed

    # M11
    M11_e = ( ax * dN1_dx * dN1_dx + ay * dN1_dy * dN1_dy )
    M11_e = sp.integrate(M11_e, (xi, 0, 1-eta))
    M11_e = -sp.integrate(M11_e, (eta, 0, 1))
    M11_e = M11_e.expand()
    M11_e = M11_e.factor(ax)
    M11_e = M11_e.factor(ay)
    M11_e = M11_e.ratsimp()

    #
    M11_e_sym = {}
    M11_e_sym[ax] = 0.0
    M11_e_sym[ay] = 0.0

    for sym in M11_e_sym.keys():
        M11_e_sym[sym] = M11_e.factor(sym)
        M11_e_sym[sym] = M11_e.coeff(sym)
        M11_e_sym[sym] = M11_e.coeff(sym).factor()
        M11_e_sym[sym] = M11_e_sym[sym].subs(x1_e-x2_e, x12_e)
        M11_e_sym[sym] = M11_e_sym[sym].subs(x2_e-x3_e, x23_e)
        M11_e_sym[sym] = M11_e_sym[sym].subs(x3_e-x1_e, x31_e)
        M11_e_sym[sym] = M11_e_sym[sym].subs(y1_e-y2_e, y12_e)
        M11_e_sym[sym] = M11_e_sym[sym].subs(y2_e-y3_e, y23_e)
        M11_e_sym[sym] = M11_e_sym[sym].subs(y3_e-y1_e, y31_e)

    print(M11_e_sym)
    
                  
    # display
    if False:
        print('shape function N1 = ', N1)
        print('shape function N2 = ', N2)
        print('shape function N3 = ', N3)
        print('x = ', x)
        print('x_xi = ', x_xi)
        print('x_eta = ', x_eta)
        print('x_const = ', x_const)
        print('y = ', y)
        print('y_xi = ', y_xi)
        print('y_eta = ', y_eta)
        print('y_const = ', y_const)
        print('J = ', J)
        print('J^-1 * det(J) = ', Jinv)
        print('det(J) = ', det_J)
        print('dN1_x_y / det(J) = ', dN1_x_y)
        print('dN2_x_y / det(J) = ', dN2_x_y)
        print('dN3_x_y / det(J) = ', dN3_x_y)
        print('M11_e = ', M11_e)

















