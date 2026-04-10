Source code for retrieving and data analysis of Jupiter's north polar mean atmoshperic state from the Juno Microwave Radiometer spectra.

`dry_adiab` and `moist_adiab` host the source code to retrieve atmos. profiles under dry and moist adiabatic endmembers respectively. 
The core file is a MCMC retrieve program sampling the atmos parameters by calling the CANOE model to build the atmosphere and run radiative transfer model.
    - `dry_adiab/redo_emcee_dryadiab_depleteNH3.py`
    - `moist_adiab/redo_emcee_moistadiab_parallel.py`

`spatial` hosts the code to generate polarmean maps of north pole.

`spectra` are codes for spectra analysis, for comparison with spectra simulated assuming a ideal dry adiabatic + uniform-mixed NH3 atmosphere, as well as with Equatorial atmospheric spectra.

`atmosprofiles` shows the retrieved atmosphere parameters and profiles.

`EZ_vs_NP` makes cross comparsion with literatures, NH3, Temperature, Radio occultation, Cassini, Voyager etc.

CANOE refer to https://github.com/chengcli/canoe.git,  code version used here is achived at https://github.com/jihenghu/canoe/tree/jh/junoemcee.

