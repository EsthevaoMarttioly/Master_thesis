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
@het(exogenous=['Pi'], policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, y, r, beta, eis):
    """Single backward iteration step using EGM.
    Va_p     : array (nE, nA), expected marginal value of assets next period
    Va       : array (nE, nA), marginal value of assets today
    a_grid   : array (nA), asset grid
    a        : array (nE, nA), asset policy today
    c        : array (nE, nA), consumption policy today"""

    c_nextgrid = (beta[:, np.newaxis] * Va_p) ** (-eis)  # u'(ct+1) = beta * E(Va^(t+1)) = c_{t+1}^(-1/eis)
    coh = (1 + r) * a_grid + y[..., np.newaxis]

    # We solve a as function of CoH, but interpolating on the grid
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    a = np.maximum(a, a_grid[0])          # a >= amin
    c = coh - a                           # c + a' = (1+r)*a + y = CoH
    Va = (1 + r) * c ** (-1/eis)          # Va^t = (1+r) * u'(c_t)

    return Va, a, c



## 2. Grid, Transition Matrices and Income
def make_grid(rho_e, sd_e, nE, amin, amax, nA,
              beta_high, dbeta, lambda_I, q, f, s):
    # The Rouwenhorst method discretize the AR(1) process for income
    e_grid, _, Pi_e = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.asset_grid(amin=amin, amax=amax, n=nA)   # Log-spaced grid for assets


    # Employment: 0=employed, 1=unemployed
    Pi_s = np.vstack(([1 - s, s],     # Pi_s[E,U] = s    (loses job)
                      [f, 1 - f]))    # Pi_s[U,E] = f    (finds job)


    # Beta grid: Impatient (beta_low) and Patient (beta_high)
    beta_low = beta_high - dbeta
    b_grid = np.array([beta_low, beta_high])
    pi_b = np.array([lambda_I, 1 - lambda_I])       # stationary: [impatient, patient]
 
    # q : prob of redrawing beta type each period (0.01 => near-permanent)
    Pi_b = (1 - q) * np.eye(2) + q * np.outer(np.ones(2), pi_b)

    # Kronecker: (beta) x (s x e)
    Pi_se = np.kron(Pi_s, Pi_e)      # (nS*nE, nS*nE)
    Pi = np.kron(Pi_b, Pi_se)        # (nBeta*nS*nE, nBeta*nS*nE)
 
    # beta vector: each state carries its type's discount factor
    beta = np.kron(b_grid, np.ones(2*nE))     # (nBeta*nS*nE,)
    return e_grid, Pi, a_grid, beta



def labor_income(e_grid, w, b, tau, Tr):
    # Set grid length
    nE = np.ones(len(e_grid))

    y_emp   = (1 - tau) * w * e_grid + Tr * nE   # [employed]
    y_unemp = b * nE                             # [unemployed]

    y = np.tile(np.r_[y_emp, y_unemp], 2)        # shape (nBeta * 2 * nE)
    return y



## 3. The employment status: 1 if unemployed (s=U), 0 if employed (s=E).
## Unemp Mass (U) = 1 - integral of 1[s = E] * dLambda(s,e,a) = 1 - L
def unemployment(c, nE):
    N    = c.shape[0]                 # nBeta * nS * nE
    rows = np.arange(N)
    mask = ((rows // nE) % 2) == 1    # True for unemployed states
    u    = np.where(mask[:, np.newaxis], 1.0, 0.0) * np.ones_like(c)
    return u



## 4. The Household Block
hh = household.add_hetinputs([make_grid, labor_income])
hh = hh.add_hetoutputs([unemployment])


print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')


# from code.parameters import calibration, unknowns_ss

# calibration["w"] = 0.9
# calibration["Tr"] = 0.05

# hh.steady_state(calibration)
