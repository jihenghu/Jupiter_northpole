#! /usr/bin/env python3
import pandas as pd
import numpy as np
import emcee
import sys, os
from multiprocessing import Pool, current_process
import threading

sys.path.append("build/python")
# sys.path.append("build/bin/")
from canoe import def_species, load_configure
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock
# from canoe.harp import radiation_band, radiation

os.environ["OMP_NUM_THREADS"] = "1"
local_storage = threading.local()

def set_atmos_run_RT_concurrent(qNH3: float, 
                     T0: float = 180.0, 
                     RHmax: float=1.0,
                     adlnNH3dlnP: float=0.0,
                     pmin: float = 0.0, 
                     pmax: float = 0.0,
                     jindex: int = 0
                     ):  
    ## construct atmos with a rh limit
    ## jindex is the index of current processer, starting from zero, will add to mb.jst in canoe backend 
    xH2O=2500.
    max_inter=500
    mb.construct_atmosphere(pin, qNH3, T0, RHmax, jindex, "dry", xH2O, max_inter)
    ## modify the top humidity with a increment
    # mb.modify_dlnNH3dlnP_rhmax(pin, adlnNH3dlnP, pmin, pmax, RHmax, jindex, "dry") 

    ## do radiative transfer
    rad.cal_radiance(mb, mb.k_st, mb.j_st+jindex)
    tb = np.array([0.0] * 4 * nb)
    for ib in range(nb):
        toa = rad.get_band(ib).get_toa()[0]
        tb[ib * 4 : ib * 4 + 4] = toa
    return tb[4:]

# Define likelihood function
def ln_likelihood(theta):
    nh3, temp = theta

    # process_id = os.getpid()  # This gives the process ID
    thread_id = current_process().name.split('-')[1]  # This gives the name of the current process
    # print(f"Task is being executed by process ID: {process_id} and thread ID: {thread_id}")
    jindex=int(thread_id)-1 ## -1 is extremely important!! do not touch it if you are not sure
    simulations = set_atmos_run_RT_concurrent(nh3, temp, 1., 0, 0, 0, jindex)  # Use your forward operator here

    # print("observations:", observations)
    # print("simulations:", simulations)
    residuals = observations - simulations

    chi_squared=0.0
    for ich in range(5): # five channels
        diff=residuals[ich*4:(ich+1)*4]
        cov=COV[ich*4:(ich+1)*4,ich*4:(ich+1)*4]
        chi_squared += np.dot(diff, np.linalg.solve(cov, diff))
    return -0.5 * chi_squared

# Define priors for NH3 and temperature
def ln_prior(theta):
    nh3, temp= theta

    nh3_mean = 300  # Mean value for NH3
    nh3_stddev = 100  # Standard deviation for NH3

    temp_mean = 169  # Mean value for temperature
    temp_stddev = 10  # Standard deviation for temperature   0.5%

 
    ln_prior_nh3 = -0.5 * ((nh3 - nh3_mean) / nh3_stddev)**2 - np.log(nh3_stddev * np.sqrt(2 * np.pi))
    ln_prior_temp = -0.5 * ((temp - temp_mean) / temp_stddev)** 2 - np.log(temp_stddev * np.sqrt(2 * np.pi))


    if (0 < nh3 < 1000) and (100 < temp < 200):
        return ln_prior_nh3 + ln_prior_temp
    return -np.inf  # return negative infinity if parameters are outside allowed range

# Combine likelihood and prior to get posterior
def ln_posterior(theta):
    prior = ln_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    return prior + ln_likelihood(theta)

if __name__=="__main__":

    ## initialize Canoe
    global pin
    pin = ParameterInput()
    pin.load_from_file("juno_mwr.inp")

    vapors = pin.get_string("species", "vapor").split(", ")
    clouds = pin.get_string("species", "cloud").split(", ")
    tracers = pin.get_string("species", "tracer").split(", ")

    def_species(vapors=vapors, clouds=clouds, tracers=tracers)
    def_thermo(pin)

    config = load_configure("juno_mwr.yaml")
    # print(pin.get_real("problem", "qH2O.ppmv"))

    pin.set_boolean("job","verbose", False)

    # pin.set_string("problem","atm_file", "build/bin/jupiter.yaml")

    nx2 = 12  ## shall not be less than N_walkers, can be a little greater for safty. should > 2 * n_variables
    print(pin.get_string("mesh","nx2"))
    pin.set_string("mesh","nx2", f"{nx2}")

    print(pin.get_string("mesh","nx2"))

    mesh = Mesh(pin)
    mesh.initialize(pin)

    global mb, rad, nb
    mb = mesh.meshblock(0)
    rad = mb.get_rad()
    nb = rad.get_num_bands()

    ##  extract TB observations from ZZ fitting results
    global observations
    observations = np.zeros((20,))
    obs = np.zeros((24,))

    mu = np.cos(np.array([0.0, 15.0, 30.0, 45.0]) / 180.0 * np.pi)
    print(mu)


    # read MWR coefficients
    mwr_coeff = "../../mwr_polar_coefficients.csv"
    df = pd.read_csv(mwr_coeff, header=None, names=["channel", "c0", "c1", "c2"])
    c0 = df["c0"].values
    c1 = df["c1"].values
    c2 = df["c2"].values

    LD0=[1.0, 10.1, 6.9, 5.9, 4.5, 1.4]  # Fabiabo crrection on LD

    # generate observations
    for ch in range(6):
         # Xr=1.0 ## \mu >0.6
        c0_=c0[ch]
        c1_=c1[ch]
        c2_=c2[ch]


        mu45=np.cos(np.radians(45.0))
        tb45=c0_-c1_*5.0*(1-mu45)+c2_/0.04*0.5*(mu45 - 0.8)*(1 - mu45)
        R45=(c0_-tb45)/c0_*100.0
        # print(f"ch{ch+1},c0, c1, c2_, R45:", c0_, c1_, c2_, R45)
        if ch>0:
           r_=LD0[ch]/R45
           c1_=c1_*r_
           c2_=c2_*r_

        print(f"ch{ch+1},c0, c1, c2_, R45:", c0_, c1_, c2_, R45)

        obs[(ch)*4:(ch+1)*4] =(c0_-c1_*5.0*(1-mu)+c2_/0.04*0.5*(mu - 0.8)*(1 - mu))

        tb45=c0_-c1_*5.0*(1-mu45)+c2_/0.04*0.5*(mu45 - 0.8)*(1 - mu45)
        R45=(c0_-tb45)/c0_*100.0
        print(R45)
    # exit()
    ## discard CH1
    observations = obs[4:]
    print(observations)   

    ## construct covariance matrix COV
    sigma=0.5 #K   random gaussian noise
    nchannel=20
    noise_var = np.zeros((nchannel, nchannel))
    # Set the diagonal values to 0.5 * 0.5
    np.fill_diagonal(noise_var, sigma**2)

    ## calibration error 2%->1%
    caliberr=0.01 
    calib_var = np.full((nchannel, nchannel), caliberr**2)
    calib_var[0:4,4:]=0.0
    calib_var[4:,:4]=0.0
    calib_var[4:8,8:]=0.0
    calib_var[8:,4:8]=0.0
    calib_var[8:12,12:]=0.0
    calib_var[12:,8:12]=0.0
    calib_var[12:16,16:]=0.0
    calib_var[16:,12:16]=0.0

    # Calculate the number of elements in the obs array
    num_obs = len(observations)

    # Create a matrix of size obs*obs filled with zeros
    TT = np.zeros((num_obs, num_obs))

    # Fill the matrix with the product of each element in obs
    for i in range(num_obs):
        for j in range(num_obs):
            TT[i][j] = observations[i] * observations[j]

    global COV
    COV=TT*calib_var+noise_var
    # print(COV)

    # proform MCMC 
    # Initialize walkers
    n_walkers = nx2
    n_dimensions = 2  # nh3, temperature

    nh3_range= (0,1000)
    temp_range = (100, 200)  


    # # Generate random initial guesses for all walkers

    # Fill in the initial_guess array with random values within the specified ranges
    # Generate random initial guesses
    initial_guesses = [
    [   np.random.uniform(nh3_range[0], nh3_range[1]),
        np.random.uniform(temp_range[0], temp_range[1]),
    ]
    for _ in range(n_walkers)
    ]

    # Run MCMC
    n_steps = 5000

    # backend
    filename = f"run_emcee_dryadiab_uniformNH3_LD101_parallel_{n_steps}.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(n_walkers, n_dimensions)

    POOL_SIZE=n_walkers

    with Pool(POOL_SIZE) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dimensions, ln_posterior, backend=backend,pool=pool)
        sampler.run_mcmc(initial_guesses, n_steps, progress=True)
