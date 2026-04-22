#=
#---------------------------------------------------------------------------
# DESCRIPTION
# Define the household block of the model, which includes the EGM problem,
# the grid and transition matrices for income, and the labor income function.
#---------------------------------------------------------------------------
#=

# Import Packages
import random
import numpy as np

from sequence_jacobian import het, interpolate, grids


# Set a seed for future replications
random.seed(20260415)


## 1. EGM Problem
## Initial guess for marginal value of assets: V_a = (1+r) * c^{-1/eis}
## Cash on Hand = (1+r)*a + y(s,e);    Arbitrary guess: c = y + r*a
def household_init(a_grid, y, r, eis):
    c = np.maximum(1e-8, y[..., np.newaxis] + np.maximum(r, 0.04) * a_grid)
    Va = (1 + r) * (c ** (-1 / eis))
    return Va


## Endogenous Grid Method (EGM) for the HH problem
@het(exogenous=['Pi_s', 'Pi_e'], policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, y, r, beta, eis):
    """Single backward iteration step using EGM.
    Va_p     : array (nE, nA), expected marginal value of assets next period
    Va       : array (nE, nA), marginal value of assets today
    a_grid   : array (nA), asset grid
    a        : array (nE, nA), asset policy today
    c        : array (nE, nA), consumption policy today
    y        : array (nE), non-finantial income, given by the producticity grid
    r        : scalar, ex-post real interest rate
    T        : scalar, lump-sum transfer from the government"""

    c_nextgrid = (beta * Va_p) ** (-eis)  # u'(ct+1) = beta * E(Va^(t+1)) = c_{t+1}^(-1/eis)
    coh = (1 + r) * a_grid + y[..., np.newaxis]

    # We solve a as function of CoH, but interpolating on the grid
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])          # a >= amin
    c = coh - a                           # c + a' = (1+r)*a + y = CoH
    Va = (1 + r) * c ** (-1/eis)          # Va^t = (1+r) * u'(c_t)

    return Va, a, c


## 2. Grid, Transition Matrices and Income
def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    # The Rouwenhorst method discretize the AR(1) process for income
    e_grid, _, Pi_e = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.asset_grid(amin=amin, amax=amax, n=nA)   # Log-spaced grid for assets
    return e_grid, Pi_e, a_grid


def search_frictions(f, s):
    """
      Pi_s[E,E] = 1-s  (keeps job)
      Pi_s[E,U] = s    (loses job)
      Pi_s[U,E] = f    (finds job)
      Pi_s[U,U] = 1-f  (stays unemployed)
    SS unemployment: u* = s / (s + f)
    """
    Pi_s = np.vstack(([1 - s, s], [f, 1 - f]))
    return Pi_s


def labor_income(e_grid, w, b, tau):
    """
    Non-financial income for each (s, e) pair.
    y[0, :] = (1 - tau) * w * e   [employed]
    y[1, :] = b                   [unemployed]
    """
    y_emp   = (1 - tau) * w * e_grid
    y_unemp = b * np.ones(len(e_grid))
    y = np.vstack((y_emp, y_unemp))     # shape (2, nE)
    return y


## 3. The employment status: 1 if unemployed (s=U), 0 if employed (s=E).
## Unemp Mass (U) = 1 - integral of 1[s = E] * dLambda(s,e,a) = 1 - L
def unemployment(c):
    u = np.zeros_like(c)  # shape (2*n_e, nA) = (14, 500)
    u[1, :] = 1.0
    return u


## 4. The Household Block
hh = household.add_hetinputs([make_grid, search_frictions, labor_income])
hh = hh.add_hetoutputs([unemployment])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')

