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
# 1D case: analytic solution
#

if True:

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


