import numpy as np
import sunode
import sunode.wrappers.as_pytensor
import pymc as pm

times = np.arange(1900,1921,1)
lynx_data = np.array([
    4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
    8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6
])
hare_data = np.array([
    30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
    27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7
])


def lotka_volterra(t, y, p):
    """Right hand side of Lotka-Volterra equation.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.
    """
    return {
        'hares': p.alpha * y.hares - p.beta * y.lynx * y.hares,
        'lynx': p.delta * y.hares * y.lynx - p.gamma * y.lynx,
    }


with pm.Model() as model:
    hares_start = pm.HalfNormal('hares_start', sigma=50)
    lynx_start = pm.HalfNormal('lynx_start', sigma=50)

    ratio = pm.Beta('ratio', alpha=0.5, beta=0.5)

    fixed_hares = pm.HalfNormal('fixed_hares', sigma=50)
    fixed_lynx = pm.Deterministic('fixed_lynx', ratio * fixed_hares)

    period = pm.Gamma('period', mu=10, sigma=1)
    freq = pm.Deterministic('freq', 2 * np.pi / period)

    log_speed_ratio = pm.Normal('log_speed_ratio', mu=0, sigma=0.1)
    speed_ratio = np.exp(log_speed_ratio)

    # Compute the parameters of the ode based on our prior parameters
    alpha = pm.Deterministic('alpha', freq * speed_ratio * ratio)
    beta = pm.Deterministic('beta', freq * speed_ratio / fixed_hares)
    gamma = pm.Deterministic('gamma', freq / speed_ratio / ratio)
    delta = pm.Deterministic('delta', freq / speed_ratio / fixed_hares / ratio)

    y_hat, _, problem, solver, _, _ = sunode.wrappers.as_pytensor.solve_ivp(
        y0={
            # The initial conditions of the ode. Each variable
            # needs to specify a PyTensor or numpy variable and a shape.
            # This dict can be nested.
            'hares': (hares_start, ()),
            'lynx': (lynx_start, ()),
        },
        params={
            # Each parameter of the ode. sunode will only compute derivatives
            # with respect to PyTensor variables. The shape needs to be specified
            # as well. It it infered automatically for numpy variables.
            # This dict can be nested.
            'alpha': (alpha, ()),
            'beta': (beta, ()),
            'gamma': (gamma, ()),
            'delta': (delta, ()),
            'extra': np.zeros(1),
        },
        # A functions that computes the right-hand-side of the ode using
        # sympy variables.
        rhs=lotka_volterra,
        # The time points where we want to access the solution
        tvals=times,
        t0=times[0],
    )

    # We can access the individual variables of the solution using the
    # variable names.
    pm.Deterministic('hares_mu', y_hat['hares'])
    pm.Deterministic('lynx_mu', y_hat['lynx'])

    sd = pm.HalfNormal('sd')
    pm.LogNormal('hares', mu=y_hat['hares'], sigma=sd, observed=hare_data)
    pm.LogNormal('lynx', mu=y_hat['lynx'], sigma=sd, observed=lynx_data)

    with model:
        idata = pm.sample(tune=100, draws=100, chains=6, cores=20,progressbar= True)

    lib = sunode._cvodes.lib
    lib.CVodeSStolerances(solver._ode, 1e-10, 1e-10)
    lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
    lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
    lib.CVodeSetMaxNumSteps(solver._ode, 5000)
    lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 5000)