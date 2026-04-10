#! /usr/bin/env python3
import numpy as np
import sys, os
import h5py
from tqdm import tqdm
from multiprocessing import Pool, current_process, shared_memory
import threading

sys.path.append("build/python")
sys.path.append(".")
from canoe import def_species, load_configure, index_map
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock
# from canoe.harp import radiation_band, radiation

os.environ["OMP_NUM_THREADS"] = "1"
local_storage = threading.local()

def set_atmos_run_RT_concurrent(theta, tbs_shm_name, SH_NH3_shm_name, RH_NH3_shm_name,SH_H2O_shm_name, RH_H2O_shm_name, nstep, n_walkers):
    # [qNH3, temperature, RHmax, adlnNH3dlnP, pmax, istep] = theta
    [qH2O, temperature, RHmax, istep] = theta
    pmin = 1.E-3
    thread_id = current_process().name.split('-')[1]
    jindex = int(thread_id) - 1
    qNH3=354.96
    # Reconnect to the shared memory blocks
    # tbs_shm = shared_memory.SharedMemory(name=tbs_shm_name)
    SH_NH3_shm = shared_memory.SharedMemory(name=SH_NH3_shm_name)
    RH_NH3_shm = shared_memory.SharedMemory(name=RH_NH3_shm_name)
    SH_H2O_shm = shared_memory.SharedMemory(name=SH_H2O_shm_name)
    RH_H2O_shm = shared_memory.SharedMemory(name=RH_H2O_shm_name)

    # tbs = np.ndarray((nstep, n_walkers, 20), dtype=np.float64, buffer=tbs_shm.buf)
    SH_NH3 = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=SH_NH3_shm.buf)
    RH_NH3 = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=RH_NH3_shm.buf)
    SH_H2O = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=SH_H2O_shm.buf)
    RH_H2O = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=RH_H2O_shm.buf)

    mb.construct_atmosphere(pin, qNH3, temperature, RHmax, jindex,"pseudo",qH2O,500)
    # mb.modify_dlnNH3dlnP_rhmax(pin,adlnNH3dlnP, pmin, pmax, RHmax, jindex,"pseudo") 

    # rad.cal_radiance(mb, mb.k_st, mb.j_st + jindex)
    # tb = np.array([0.0] * 4 * nb)
    # for ib in range(nb):
    #     toa = rad.get_band(ib).get_toa()[0]
    #     tb[ib * 4 : ib * 4 + 4] = toa
    # tbs[int(istep), jindex, :] = tb[4:]

    aircolumn = mb.get_aircolumn(mb.k_st, mb.j_st + jindex, mb.i_st, mb.i_ed)   
    for i in range(len(aircolumn)):
        ap = aircolumn[i]
        SH_NH3[int(istep), jindex, i] = ap.hydro()[iNH3]
        SH_H2O[int(istep), jindex, i] = ap.hydro()[iH2O]

        ap_mole = ap.to_mole_fraction()
        RH_NH3[int(istep), jindex, i] = ap_mole.hydro()[iNH3]*1.E6  # ppmv
        RH_H2O[int(istep), jindex, i] = ap_mole.hydro()[iH2O]*1.E6  # ppmv

        # RH_NH3[int(istep), jindex, i] = ap_mole.get_rh(iNH3)
        # RH_H2O[int(istep), jindex, i] = ap_mole.get_rh(iH2O)

    # tbs_shm.close()
    SH_NH3_shm.close()
    RH_NH3_shm.close()
    SH_H2O_shm.close()
    RH_H2O_shm.close()

nx2 = 8  # Shall not be less than n_walkers, can be a little greater for safety.
global pin
pin = ParameterInput()
pin.load_from_file("juno_mwr.inp")

vapors = pin.get_string("species", "vapor").split(", ")
clouds = pin.get_string("species", "cloud").split(", ")
tracers = pin.get_string("species", "tracer").split(", ")

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
print(iNH3)

h5 = h5py.File('redo_emcee_moistadiab_FabianoLD_parallel_10000.h5', 'r')
chain = np.array(h5['mcmc']['chain'][5000:])  # [nstep, n_walkers, ndim]
h5.close()

[nstep, n_walkers, ndim] = chain.shape

# Create shared memory arrays
# tbs_shm = shared_memory.SharedMemory(create=True, size=nstep * n_walkers * 20 * np.float64().nbytes)
# tbs = np.ndarray((nstep, n_walkers, 20), dtype=np.float64, buffer=tbs_shm.buf)
# tbs.fill(0)

SH_NH3_shm = shared_memory.SharedMemory(create=True, size=nstep * n_walkers * 1600 * np.float64().nbytes)
SH_NH3 = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=SH_NH3_shm.buf)
SH_NH3.fill(0)

RH_NH3_shm = shared_memory.SharedMemory(create=True, size=nstep * n_walkers * 1600 * np.float64().nbytes)
RH_NH3 = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=RH_NH3_shm.buf)
RH_NH3.fill(0)

SH_H2O_shm = shared_memory.SharedMemory(create=True, size=nstep * n_walkers * 1600 * np.float64().nbytes)
SH_H2O = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=SH_H2O_shm.buf)
SH_H2O.fill(0)

RH_H2O_shm = shared_memory.SharedMemory(create=True, size=nstep * n_walkers * 1600 * np.float64().nbytes)
RH_H2O = np.ndarray((nstep, n_walkers, 1600), dtype=np.float64, buffer=RH_H2O_shm.buf)
RH_H2O.fill(0)


POOL_SIZE = n_walkers
thetas = np.zeros((n_walkers, 4))
with Pool(POOL_SIZE) as pool:
    for istep in tqdm(range(nstep)):
        for iw in range(n_walkers):
            thetas[iw] = [chain[istep, iw, 0], chain[istep, iw, 1], chain[istep, iw, 2], istep]
        pool.starmap(set_atmos_run_RT_concurrent, [(theta, 0, SH_NH3_shm.name, RH_NH3_shm.name, SH_H2O_shm.name, RH_H2O_shm.name, nstep, n_walkers) for theta in thetas])


# H5OUT = h5py.File(f'run_juno_emcee_H2O_rk1_adapt_TB_{nstep}.h5', 'w')
# H5OUT.create_dataset('tb', data=tbs)
# H5OUT.close()

profOUT = h5py.File(f'reconstruct_atmos_profiles_moist_last{nstep}.h5', 'w')
profOUT.create_dataset('SH_NH3', data=SH_NH3)
profOUT.create_dataset('NH3_ppmv', data=RH_NH3)
profOUT.create_dataset('SH_H2O', data=SH_H2O)
profOUT.create_dataset('H2O_ppmv', data=RH_H2O)
profOUT.close()
# Cleanup shared memory
# tbs_shm.close()
# tbs_shm.unlink()

SH_NH3_shm.close()
SH_NH3_shm.unlink()
RH_NH3_shm.close()
RH_NH3_shm.unlink()

SH_H2O_shm.close()
SH_H2O_shm.unlink()
RH_H2O_shm.close()
RH_H2O_shm.unlink()