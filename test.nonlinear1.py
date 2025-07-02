# %matplotlib inline
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano

from pymc3.ode import DifferentialEquation
from scipy.integrate import odeint
import os
# one of
# os.environ['MKL_THREADING_LAYER'] = 'sequential'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_THREADING_LAYER'] = 'GNU'

# plt.style.use("seaborn-darkgrid")

# print('*** Start script ***')
# print(f'{pm.__name__}: v. {pm.__version__}')
# print(f'{theano.__name__}: v. {theano.__version__}')
cxxflags = []
cxxflags.append('-fno-asynchronous-unwind-tables')
# freeze_support()
def SIR(y, t, p):
    ds = -p[0] * y[0] * y[1]
    di = p[0] * y[0] * y[1] - p[1] * y[1]
    return [ds, di]

if __name__ == '__main__':
    # For reproducibility
    rseed = 20394
    np.random.seed(rseed)

    times = np.arange(0, 5, 0.25)

    beta, gamma = 4, 1.0
    # Create true curves
    y = odeint(SIR, t=times, y0=[0.99, 0.01], args=((beta, gamma),), rtol=1e-8)
    # Observational model.  Lognormal likelihood isn't appropriate, but we'll do it anyway
    yobs = np.random.lognormal(mean=np.log(y[1::]), sigma=[0.2, 0.3])

    # plt.plot(times[1::], yobs, marker="o", linestyle="none")
    # plt.plot(times, y[:, 0], color="C0", alpha=0.5, label=f"$S(t)$")
    # plt.plot(times, y[:, 1], color="C1", alpha=0.5, label=f"$I(t)$")
    # plt.legend()
    # plt.show()

    sir_model = DifferentialEquation(
        func=SIR,
        times=np.arange(0.25, 5, 0.25),
        n_states=2,
        n_theta=2,
        t0=0,
    )

    with pm.Model() as model4:
        sigma = pm.HalfCauchy("sigma", 1, shape=2)

        # R0 is bounded below by 1 because we see an epidemic has occurred
        R0 = pm.Bound(pm.Normal, lower=1)("R0", 2, 3)
        lam = pm.Lognormal("lambda", pm.math.log(2), 2)
        beta = pm.Deterministic("beta", lam * R0)

        sir_curves = sir_model(y0=[0.99, 0.01], theta=[beta, lam])

        Y = pm.Lognormal("Y", mu=pm.math.log(sir_curves), sigma=sigma, observed=yobs)

        trace = pm.sample(2000, chains=2, tune=1000, target_accept=0.9, cores=6, random_seed=20394)
        data = az.from_pymc3(trace=trace)

        # az.plot_posterior(data, round_to=2, credible_interval=0.95)