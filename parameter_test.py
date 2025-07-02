import pickle
from init_param import init_pars
import numpy as np


################----打印参数----###################
params = np.array([init_pars["PRest"], init_pars["PK"], init_pars["PL"], init_pars["Kbile"], init_pars["GFR"], \
             init_pars["Free"], init_pars["Vmax_baso"], init_pars["Km_baso"], init_pars["Kurine"], init_pars["Kreab"]])
print("init_params:", params)

with open('saved_result/optimized_params.pkl', 'rb') as f:
    fit_params = pickle.load(f)
    print("fit_params:", fit_params)

MCMC_params = np.load('saved_result/best_params.npy')
print("MCMC_params:", MCMC_params)