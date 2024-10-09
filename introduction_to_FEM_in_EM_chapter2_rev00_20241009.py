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
    poisson_lhs  = Dx.diff(x, 1) + Dy.diff(y, 1)
    poisson_rhs  = rho

    # display
    print(poisson_lhs)
    print(poisson_rhs)


#==================================================
# 1.2 DOMAIN DISTRETIZATION
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
# 1.3 INTERPOLATION FUNCTIONS
#
# proper interpolation functions
#
# 1) guarantee continuity of the primary unknown quantitiy across interelement boundaries
# 2) at least once differentiable, must be complete polynomials
#    Lagrange polynomials
#
# linear triangular element
# the nodes are locally numbered in a CCW direction (Jacobian definition)
#
# x y plane -> xi(ξ) eta(η) plane
#

if True:

    # symbols
    x, y = sp.symbols('x, y')               # position
    xi, eta = sp.symbols('ξ, η')            # position
    c1, c2, c3 = sp.symbols('c1, c2, c3')
    
    # shape function N1
    N1_xieta = c1 + c2 * xi + c3 * eta
    
    N1_xieta_node1 = N1_xieta.subs({xi:0.0, eta:0.0}) - 1.0
    N1_xieta_node2 = N1_xieta.subs({xi:1.0, eta:0.0})
    N1_xieta_node3 = N1_xieta.subs({xi:0.0, eta:1.0})

    N1_xieta_eqs = [N1_xieta_node1, N1_xieta_node2, N1_xieta_node3]
    N1_xieta_sol = sp.solve(N1_xieta_eqs, [c1, c2, c3], dict=True)

    for item in N1_xieta_sol[0].keys():
        N1_xieta = N1_xieta.subs(item, N1_xieta_sol[0][item])

    # shape function N2
    N2_xieta = c1 + c2 * xi + c3 * eta
    
    N2_xieta_node1 = N2_xieta.subs({xi:0.0, eta:0.0})
    N2_xieta_node2 = N2_xieta.subs({xi:1.0, eta:0.0}) - 1.0
    N2_xieta_node3 = N2_xieta.subs({xi:0.0, eta:1.0})

    N2_xieta_eqs = [N2_xieta_node1, N2_xieta_node2, N2_xieta_node3]
    N2_xieta_sol = sp.solve(N2_xieta_eqs, [c1, c2, c3], dict=True)

    for item in N2_xieta_sol[0].keys():
        N2_xieta = N2_xieta.subs(item, N2_xieta_sol[0][item])

    # shape function N3
    N3_xieta = c1 + c2 * xi + c3 * eta
    
    N3_xieta_node1 = N3_xieta.subs({xi:0.0, eta:0.0})
    N3_xieta_node2 = N3_xieta.subs({xi:1.0, eta:0.0})
    N3_xieta_node3 = N3_xieta.subs({xi:0.0, eta:1.0}) - 1.0

    N3_xieta_eqs = [N3_xieta_node1, N3_xieta_node2, N3_xieta_node3]
    N3_xieta_sol = sp.solve(N3_xieta_eqs, [c1, c2, c3], dict=True)

    for item in N3_xieta_sol[0].keys():
        N3_xieta = N3_xieta.subs(item, N3_xieta_sol[0][item])
       
    # print
    print('N1_xieta = ', N1_xieta)
    print('N2_xieta = ',N2_xieta)
    print('N3_xieta = ',N3_xieta)
    print('')


#
# arbitrary point (ξ, η) inside the master triangle
#

if True:

    # symbols
    x1_e, x2_e, x3_e = sp.symbols('x1_e, x2_e, x3_e')
    y1_e, y2_e, y3_e = sp.symbols('y1_e, y2_e, y3_e')
    
    # primary unknown quantity
    x = x1_e *  N1_xieta + x2_e *  N2_xieta + x3_e *  N3_xieta
    y = y1_e *  N1_xieta + y2_e *  N2_xieta + y3_e *  N3_xieta

    x = x.expand()
    y = y.expand()

    #
    x_xi = x.coeff(xi)
    x_eta = x.coeff(eta)
    x_const = x - x_xi * xi - x_eta * eta
    x_const = x_const.simplify()

    #
    y_xi = y.coeff(xi)
    y_eta = y.coeff(eta)
    y_const = y - y_xi * xi - y_eta * eta
    y_const = y_const.simplify()
    
    #
    print('x_const = ', x_const)
    print('x_xi coeff = ', x_xi)
    print('x_eta coeff = ', x_eta)
    print('')
    print('y_const = ', y_const)
    print('y_xi coeff = ', y_xi)
    print('y_eta coeff = ', y_eta)
    print('')


#
# bilinear quadrilateral element
#

if True:

    # symbols


