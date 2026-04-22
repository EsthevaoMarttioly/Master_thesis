#=
#---------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid and transition matrices for income, and the labor income function.
#---------------------------------------------------------------------------
#=

# Import Packages
import numpy as np
from sequence_jacobian import simple


#---------------------------------------------------------------------------
# Firm Block:
# Production: Y = Z * L  =>  L = Y / Z
@simple
def firm(Y, Z):
    L = Y / Z
    return L


# Phillips Curve:
# log(1+pi) = kappa * (w/Z - 1/mu) + Y(+1)/Y * log(1+pi(+1)) / (1+r(+1))
# pi is a GE unknown, solved by the NKPC block
@simple
def pricing(pi, w, Z, Y, r, kappa, mu):
    nkpc = (kappa * (w / Z - 1 / mu)
            + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
            - (1 + pi).apply(np.log))
    return nkpc



#---------------------------------------------------------------------------
# Government Block
# Budget constraint:   b_t * U_t + (1+r_{t-1})*B_{t-1} = tau*w_t*L_t + B_t
@simple
def fiscal(r, w, L, U, B, tau, b):
    # At SS (B constant):  T = tau*w*L - r*B - b*U
    Tax = tau * w * L
    BenefCost = b * U
    Tr = Tax - r * B - BenefCost
    return Tax, BenefCost, Tr


# Monetary Policy
# Taylor rule:   i = rstar(-1) + phi * pi(-1)  +  Fisher Equation
@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r



#---------------------------------------------------------------------------
# Market Clearing
@simple
def mkt_clearing(A, C, L, Y, B, U):
    asset_mkt = A - B
    labor_mkt = (1 - U) - L
    goods_mkt = Y - C
    return asset_mkt, labor_mkt, goods_mkt

