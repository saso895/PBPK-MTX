#https://num.pyro.ai/en/stable/tutorials/lotka_volterra_multiple.html
import functools

import matplotlib.pyplot as plt

import jax
from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample

# Numerical instabilities may arise during ODE solving,
# so one has sometimes to play around with solver settings,
# change solver, or change numeric precision as we do here.
numpyro.enable_x64(True)

def dz_dt(z, t, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describes the interaction of two species.
    """
    u, v = z
    alpha, beta, gamma, delta = theta

    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


def model(ts, y_init, y=None):
    """
    :param numpy.ndarray ts: measurement times
    :param numpy.ndarray y_init: measured inital conditions
    :param numpy.ndarray y: measured populations
    """
    # initial population
    z_init = numpyro.sample(
        "z_init", dist.LogNormal(jnp.log(y_init), jnp.ones_like(y_init))
    )

    # parameters alpha, beta, gamma, delta of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0,
            loc=jnp.array([1.0, 0.05, 1.0, 0.05]),
            scale=jnp.array([0.2, 0.01, 0.2, 0.01]),
        ),
    )

    # helpers to solve ODEs in a vectorized form
    odeint_with_kwargs = functools.partial(odeint, rtol=1e-6, atol=1e-5, mxstep=1000)
    vect_solve_ode = jax.vmap(
        odeint_with_kwargs,
        in_axes=(None, 0, 0, None),
    )

    # integrate dz/dt
    zs = vect_solve_ode(dz_dt, z_init, ts, theta)
    # measurement errors
    sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([2]))
    # measured populations
    if y is not None:
        # mask missing observations in the observed y
        mask = jnp.isfinite(jnp.log(y))
        numpyro.sample("y", dist.LogNormal(jnp.log(zs), sigma).mask(mask), obs=y)
    else:
        numpyro.sample("y", dist.LogNormal(jnp.log(zs), sigma))

n_datasets = 3  # int n_datasets: number of datasets to generate
t_min = 100  # int t_min: minimal allowed length of the generated time array
t_max = 200  # int t_min: maximal allowed length of the generated time array
n_points_min = 80  # int n_points_min: minimal allowed number of points in a data set
n_points_max = 120  # int n_points_max: maximal allowed number of points in a data set
y0_min = 2.0  # float y0_min: minimal allowed value for initial conditions
y0_max = 10.0  # float y0_max: maximal allowed value for initial conditions
p_missing = 0.1  # float p_missing: probability of having missing values

# generate an array with initial conditons
z_inits = jnp.array(
    [jnp.linspace(y0_min, y0_max, n_datasets), jnp.linspace(y0_max, y0_min, n_datasets)]
).T

print(f"Initial conditons are: \n {z_inits}")

# generate array with random integers between t_min and t_max, representing tiem duration in the data set
rand_duration = jax.random.randint(
    PRNGKey(1), shape=(n_datasets,), minval=t_min, maxval=t_max
)

# generate array with random integers between n_points_min and n_points_max,
# representing number of time points per dataset
rand_n_points = jax.random.randint(
    PRNGKey(1), shape=(n_datasets,), minval=n_points_min, maxval=n_points_max
)

# Note that arrays have different length and are stored in a list
time_arrays = [
    jnp.linspace(0, j, num=rand_n_points[i]).astype(float)
    for i, j in enumerate(rand_duration)
]
longest = jnp.max(jnp.array([len(i) for i in time_arrays]))

# Make a time matrix
ts = jnp.array(
    [
        jnp.pad(arr, pad_width=(0, longest - len(arr)), constant_values=jnp.nan)
        for arr in time_arrays
    ]
)

print(f"The shape of the time matrix is {ts.shape}")
print(f"First values are \n {ts[:, :10]}")
print(f"Last values are \n {ts[:, -10:]}")

# take a single sample that will be our synthetic data
sample = Predictive(model, num_samples=1)(PRNGKey(100), ts, z_inits)
data = sample["y"][0]

# create a mask that will add missing values to the data
missing_obs_mask = jax.random.choice(
    PRNGKey(1),
    jnp.array([True, False]),
    shape=data.shape,
    p=jnp.array([p_missing, 1 - p_missing]),
)
# make sure that initial values are not missing
missing_obs_mask = missing_obs_mask.at[:, 0, :].set(False)

# data with missing values
data = data.at[missing_obs_mask].set(jnp.nan)

# fill_nans
def fill_nans(ts):
    n_nan = jnp.sum(jnp.isnan(ts))
    if n_nan > 0:
        loc_first_nan = jnp.where(jnp.isnan(ts))[0][0]
        ts_filled_nans = ts.at[loc_first_nan:].set(
            jnp.linspace(t_max, t_max + 20, n_nan)
        )
        return ts_filled_nans
    else:
        return ts


ts_filled_nans = jnp.array([fill_nans(t) for t in ts])

print(f"The dataset has the shape {data.shape}, (n_datasets, n_points, n_observables)")
print(f"The time matrix has the shape {ts.shape}, (n_datasets, n_timepoints)")
print(f"The time matrix has different spacing between timepoints: \n {ts[:,:5]}")
print(f"The final timepoints are: {jnp.nanmax(ts,1)} years.")
print(
    f"The dataset has {jnp.sum(jnp.isnan(data))/jnp.size(data):.0%} missing observations"
)
print(f"True params mean: {sample['theta'][0]}")

# Plotting
fig, axs = plt.subplots(2, n_datasets, figsize=(15, 4))

for i in range(n_datasets):
    loc = jnp.where(jnp.isfinite(data[i, :, 0]))[0][-1]

    axs[0, i].plot(
        ts[i, :], data[i, :, 0], "ko", mfc="none", ms=4, label="true hare", alpha=0.67
    )
    axs[0, i].plot(ts[i, :], data[i, :, 0], label="true hare", alpha=0.67)
    axs[0, i].set_xlabel("Time, year")
    axs[0, i].set_ylabel("Population")
    axs[0, i].set_xlim([-5, jnp.nanmax(ts)])

    axs[1, i].plot(ts[i, :], data[i, :, 1], "bx", label="true lynx")
    axs[1, i].plot(ts[i, :], data[i, :, 1], label="true lynx")
    axs[1, i].set_xlabel("Time, year")
    axs[1, i].set_ylabel("Population")
    axs[1, i].set_xlim([-5, jnp.nanmax(ts)])

fig.tight_layout()

y_init = data[:, 0, :]

mcmc = MCMC(
    NUTS(
        model,
        dense_mass=True,
        init_strategy=init_to_sample(),
        max_tree_depth=10,
    ),
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    progress_bar=True,
)

mcmc.run(PRNGKey(1031410), ts=ts_filled_nans, y_init=y_init, y=data)
mcmc.print_summary()

print(f"True params mean: {sample['theta'][0]}")
print(f"Estimated params mean: {jnp.mean(mcmc.get_samples()['theta'], axis = 0)}")

# predict
ts_pred = jnp.tile(jnp.linspace(0, 200, 1000), (n_datasets, 1))
pop_pred = Predictive(model, mcmc.get_samples())(PRNGKey(1041140), ts_pred, y_init)["y"]
mu = jnp.mean(pop_pred, 0)
pi = jnp.percentile(pop_pred, jnp.array([10, 90]), 0)


print(f"True params mean: {sample['theta'][0]}")
print(f"Estimated params mean: {jnp.mean(mcmc.get_samples()['theta'], axis = 0)}")

# Plotting
fig, axs = plt.subplots(2, n_datasets, figsize=(15, 4))

for i in range(n_datasets):
    loc = jnp.where(jnp.isfinite(data[i, :, 0]))[0][-1]

    axs[0, i].plot(
        ts_pred[i, :], mu[i, :, 0], "k-.", label="pred hare", lw=1, alpha=0.67
    )
    axs[0, i].plot(
        ts[i, :], data[i, :, 0], "ko", mfc="none", ms=4, label="true hare", alpha=0.67
    )
    axs[0, i].fill_between(
        ts_pred[i, :], pi[0, i, :, 0], pi[1, i, :, 0], color="k", alpha=0.2
    )
    axs[0, i].set_xlabel("Time, year")
    axs[0, i].set_ylabel("Population")
    axs[0, i].set_xlim([-5, jnp.nanmax(ts)])

    axs[1, i].plot(ts_pred[i, :], mu[i, :, 1], "b--", label="pred lynx")
    axs[1, i].plot(ts[i, :], data[i, :, 1], "bx", label="true lynx")
    axs[1, i].fill_between(
        ts_pred[i, :], pi[0, i, :, 1], pi[1, i, :, 1], color="b", alpha=0.2
    )
    axs[1, i].set_xlabel("Time, year")
    axs[1, i].set_ylabel("Population")
    axs[1, i].set_xlim([-5, jnp.nanmax(ts)])


fig.tight_layout()

