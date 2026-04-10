#! /usr/bin/env python3
import numpy as np
import emcee
import sys, os
import matplotlib.pyplot as plt
import h5py
from scipy.stats import norm
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, current_process, shared_memory
import threading
import queue
import time

sys.path.append("build/python")
sys.path.append(".")
from canoe import def_species, load_configure, index_map
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock
# from canoe.harp import radiation_band, radiation

os.environ["OMP_NUM_THREADS"] = "1"
local_storage = threading.local()

def set_atmos_run_RT_concurrent(theta, air_temp_shm_name,air_theta_shm_name, nstep, n_walkers):
    [qNH3, temperature, RHmax, adlnNH3dlnP, pmax, istep] = theta
    # [qH2O,qNH3, temperature, RHmax, istep] = theta

    thread_id = current_process().name.split('-')[1]
    jindex = int(thread_id) - 1

    # Reconnect to the shared memory blocks
    air_temp_shm = shared_memory.SharedMemory(name=air_temp_shm_name)
    air_theta_shm = shared_memory.SharedMemory(name=air_theta_shm_name)
    air_temp = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=air_temp_shm.buf)
    air_theta = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=air_theta_shm.buf)

    mb.construct_atmosphere(pin, qNH3, temperature, RHmax, jindex,"dry",2500,500)

    # mb.construct_atmosphere(pin, 381.6, temperature, RHmax, jindex,"pseudo",qH2O,500)
    
    aircolumn = mb.get_aircolumn(mb.k_st, mb.j_st + jindex, mb.i_st, mb.i_ed)   
    for i in range(len(aircolumn)):
        ap = aircolumn[i]
        # SH_NH3[int(istep), jindex, i] = ap.hydro()[iNH3]
        # print(ap.hydro()[0])
        # ap_mole = ap.to_mole_fraction()
        # air_temp[int(istep), jindex, i] = ap_mole.get_rh(iNH3)
        # air_temp[int(istep), jindex, i] = ap_mole.hydro()[0]
        air_temp[int(istep), jindex, i] = mb.get_temp(mb.k_st, mb.j_st + jindex,i)
        # print(air_temp[int(istep), jindex, i])
        
        air_theta[int(istep), jindex, i] = mb.get_theta(P0, mb.k_st, mb.j_st + jindex,i)
        # print(air_theta[int(istep), jindex, i])

    air_temp_shm.close()
    air_theta_shm.close()

nx2 = 12  # Shall not be less than n_walkers, can be a little greater for safety.
global pin,P0
pin = ParameterInput()
pin.load_from_file("juno_mwr.inp")

vapors = pin.get_string("species", "vapor").split(", ")
clouds = pin.get_string("species", "cloud").split(", ")
tracers = pin.get_string("species", "tracer").split(", ")
P0 = pin.get_real("mesh", "ReferencePressure")

# print(P0)
# exit()
def_species(vapors=vapors, clouds=clouds, tracers=tracers)
def_thermo(pin)

config = load_configure("juno_mwr.yaml")

pin.set_boolean("job", "verbose", False)
pin.set_string("mesh", "nx2", f"{nx2}")

mesh = Mesh(pin)
mesh.initialize(pin)

global mb, rad, nb
mb = mesh.meshblock(0)
rad = mb.get_rad()
nb = rad.get_num_bands()

global iNH3,iH2O
pindex = index_map.get_instance()
iNH3 = pindex.get_vapor_id("NH3")
iH2O = pindex.get_vapor_id("H2O")
# print(iNH3)

h5 = h5py.File('redo_emcee_dryadiab_depletedNH3_FabianoLD_101_parallel_10000.h5', 'r')
chain = np.array(h5['mcmc']['chain'][5000:])  # [nstep, n_walkers, ndim]
h5.close()
[nstep, n_walkers, ndim] = chain.shape

# Create shared memory arrays

air_temp_shm = shared_memory.SharedMemory(create=True, size=nstep * n_walkers * 1600 * np.float64().nbytes)
air_theta_shm = shared_memory.SharedMemory(create=True, size=nstep * n_walkers * 1600 * np.float64().nbytes)
air_temp = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=air_temp_shm.buf)
air_theta = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=air_theta_shm.buf)
air_temp.fill(0)
air_theta.fill(0)

POOL_SIZE = n_walkers
thetas = np.zeros((n_walkers, 6))
# thetas = np.zeros((n_walkers, 4))
with Pool(POOL_SIZE) as pool:
    for istep in tqdm(range(nstep)):
        for iw in range(n_walkers):
            thetas[iw] = [chain[istep, iw, 0], chain[istep, iw, 1], chain[istep, iw, 2], chain[istep, iw, 3], chain[istep, iw, 4], istep]
            # thetas[iw] = [chain[istep, iw, 0], chain[istep, iw, 1], chain[istep, iw, 2], istep]
        pool.starmap(set_atmos_run_RT_concurrent, [(theta, air_temp_shm.name,air_theta_shm.name, nstep, n_walkers) for theta in thetas])

profOUT = h5py.File(f'reconstruct_temp_prof_dry_last{nstep}_LD101.h5', 'w')

profOUT.create_dataset('Temperature', data=air_temp)
profOUT.create_dataset('PotentialTemperature', data=air_theta)
profOUT.close()
# Cleanup shared memory

air_temp_shm.close()
air_temp_shm.unlink()
air_theta_shm.close()
air_theta_shm.unlink()
