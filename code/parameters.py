#=
#---------------------------------------------------------------------------
# DESCRIPTION
# Define the parameters of the model,
# including calibration values and unknowns to be estimated.
#---------------------------------------------------------------------------
#=

# Calibration values
calibration = dict(
    # Households
    eis   = 0.5,     # elasticity of intertemporal substitution
    # Labor market
    f     = 0.4,     # job-finding probability
    s     = 0.1,     # separation probability  =>  U_ss = s/(s+f) = 0.2
    # Productivity process (Rouwenhorst)
    rho_e = 0.966,
    sd_e  = 0.5,
    nE    = 7,
    # Asset grid
    amin  = 0.0,
    amax  = 200.0,
    nA    = 500,
    # Prices (SS targets/normalizations)
    Z     = 1.0,     # TFP (normalized)
    pi    = 0.0,     # inflation = 0 at SS
    r     = 0.02,    # real interest rate at SS
    # Government
    tau   = 0.25,    # flat labor income tax rate
    b     = 0.1,     # unemployment benefit
    B     = 1.0,     # govt bond supply
    # Monetary
    phi   = 1.5,     # Taylor rule coefficient on inflation
    rstar = 0.02,    # SS neutral real rate
    # Firms
    mu    = 1.11,    # price markup
    kappa = 0.1,     # NKPC slope (Rotemberg adjustment cost)
)


# Unknown values to be estimated, with initial guess
unknowns_ss = dict(
    beta = 0.96,    # discount factor
    w    = 0.7,     # real wage - must be 0.90 to match Z/mu
    Y    = 1.0,     # output
)

