"""
This file is responsible for loading the hydrodynamical data, and running the particle simulation from the config file. 
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import parcels
import toml
import xarray as xr
from parcels import Field, FieldSet, ParticleSet
from tqdm import tqdm

from parcels_custom import beaching_mappings
from delft_to_parcels_tooling import delft_to_fieldset
from tooling import von_neuman
from env import (CONFIG_FOLDER, HYDRO_FOLDER, PARTICLE_SIM_FOLDER,
                 PARTICLE_PLOT_FOLDER, TEMPLATE_FOLDER)
from __init__ import logger

# %%
def run_particles_from_cfg(cfg: dict, force_rerun: bool = False):
    """
    Runs the particle simulation from the config dict. Skips the simulation if the trajectory output is already cached.
    """
    # ================
    # Defining paths
    # ================
    hydro_sim_folder = HYDRO_FOLDER / cfg["hydro_model"]["ID_human"]
    particle_sim_folder = PARTICLE_SIM_FOLDER / cfg["particle_model"]["ID_human"]
    trajectories_out = particle_sim_folder / "trajectories.nc"
    particle_init_plot = PARTICLE_PLOT_FOLDER / f"{cfg['toml_fname'].split('.')[0]}.png" # Location particle initialisation plot is saved

    # ================
    # Checking if the simulation has already been run
    # ================
    if not Path(trajectories_out).exists() or force_rerun:
        logger.info(f"Running simulation {cfg['ID_human']} from toml file {cfg['toml_fname']}")
    else:
        logger.info(f"Skipping simulation {cfg['ID_human']} from toml file {cfg['toml_fname']}. Output already exists.")
        return

    logger.info("Setting up model")

    # ================
    # Loading fieldset
    # ================
    field_snapshot_path = hydro_sim_folder / "field_snapshot.nc"
    if field_snapshot_path.exists():
        ds_flow = xr.load_dataset(field_snapshot_path)
    else:
        ds_flow = xr.load_dataset(hydro_sim_folder / "trim-simulation.nc")
        ds_flow = ds_flow.isel(time=-1)
        ds_flow.to_netcdf(field_snapshot_path) # Saving the field for future use

    ds_flow = delft_to_fieldset(ds = ds_flow, return_ds=True)

    variables = {
        "U": "U",
        "V": "V",
        }
    dimensions = {
            "lon": "x_mesh", # XCOR dimension corresponding to the `MC` index in the netcdf file
            "lat": "y_mesh", # YZ dimension corresponding to the `N` index in the netcdf file
            # "depth": "KMAXOUT_RESTR", # Needs to be gridded
    }

    # TRYING TO DODGE THE MEMORY ACCESS VIOLATION AS FIRST MENTIONED HERE https://github.com/OceanParcels/parcels/issues/793
    # BY READING IN FROM FILE INSTEAD OF XARRAY
    flow_snapshot = hydro_sim_folder / "flow_snapshot.nc"
    if not flow_snapshot.exists():
        ds_flow.to_netcdf(flow_snapshot)

    fieldset = FieldSet.from_netcdf(flow_snapshot, variables, dimensions, mesh="flat", interp_method="cgrid_velocity", gridindexingtype="mitgcm", allow_time_extrapolation=True)

    # Adding land field
    ds_land = xr.load_dataset(hydro_sim_folder / "land_mask.nc")
    ds_land = ds_land.rename({"lon":"x", "lat":"y"})
    ds_flow = ds_flow.swap_dims({"x_index": "x", "y_index": "y"}) # Creating fieldset equivalent in xarray


    # Defining spawning region to be within (spawning_band)km of the land.
    # ie. within the neighbouring (spawning_band/dx) cells of the land (von Neumann Neighbourhood).
    land_array = ds_land.land.astype(bool).values
    spawning_array = von_neuman(land_array, n = int(cfg["particle_model"]["spawning_band"]/cfg["coastline"]["dx"]) + 1) & ~von_neuman(land_array, 1) # ie. can spawn anywhere within x km from the beaching region. Can't spawn in the beaching region itself.

    ds_land = ds_land.assign(particle_spawning = xr.DataArray(spawning_array.astype(float), dims = ["y", "x"]))
    land = Field.from_xarray(ds_land.land, "land", { # Defines the field that will be used for land masking
        "lon": "x",
        "lat": "y"
    }, mesh = "flat")
    particle_spawning = Field.from_xarray(ds_land.particle_spawning, "particle_spawning", { # Defines the field that will be used for particle spawning
        "lon": "x",
        "lat": "y"
    }, mesh = "flat")

    # Creating xarray dataset for fieldset
    ds_fieldset = xr.merge([ds_flow, ds_land])

    # ================
    # Modifying fieldset
    # ================
    fieldset.add_field(land)
    fieldset.add_field(particle_spawning)
    kh_value = cfg["particle_model"]["K_h"]
    fieldset.add_constant_field("Kh_zonal", kh_value, mesh="flat")
    fieldset.add_constant_field("Kh_meridional", kh_value, mesh="flat")


    # ================
    # Spawning particles based on spawning field
    # ================
    n_particles = cfg["particle_model"]["n_particles"]
    np.random.seed(cfg["particle_model"]["particle_seed"])
    lons, lats = ParticleSet.monte_carlo_sample(fieldset.particle_spawning, size = n_particles, mode="monte_carlo")


    # ================
    # Beaching related fieldset transforms, getting pset and kernel
    # ================
    beaching_spec = beaching_mappings[cfg["particle_model"]["beaching"]["beaching_key"]]
    fieldset_options = cfg["particle_model"]["beaching"]["fieldset_options"]
    fieldset = beaching_spec.get_fieldset(fieldset, **fieldset_options)
    pset = ParticleSet(fieldset=fieldset, pclass=beaching_spec.get_particle(), lon=lons, lat=lats)
    pset.show(field=fieldset.land, savefile = f"{particle_init_plot}") # Saving visualisations of final particle positions

    # Create a ParticleFile object to store the output
    particle_sim_folder.mkdir(exist_ok=True, parents=True)
    output_file = pset.ParticleFile(name=str(trajectories_out), outputdt=timedelta(minutes=30))

    # ================
    # Run simulation
    # ================
    logger.info("Running simulation")
    pset.execute(
        beaching_spec.get_kernel(pset),
        runtime=timedelta(days=cfg["particle_model"]["duration_days"]), # runtime controls the interval of the plots
        dt=timedelta(minutes=cfg["particle_model"]["dt_mins"]),
        recovery=beaching_spec.recovery,
        output_file=output_file,
    )  # the recovery kernel
    output_file.export()
    logger.success(f"Simulation complete and saved to: {trajectories_out}")
    ds_fieldset.to_netcdf(particle_sim_folder / "fieldset.nc")
    logger.success(f"Saved fieldset to: {particle_sim_folder / 'fieldset.nc'}")
    return

if __name__ == "__main__":
    configs_to_run = [
        "concave_1km_lebreton.toml",
        "concave_1km_mheen.toml",
        "concave_1km_naive.toml",
        "concave_1km_oninkb.toml",
        "concave_1km_oninkbr.toml",
        "convex_1km_lebreton.toml",
        "convex_1km_mheen.toml",
        "convex_1km_naive.toml",
        "convex_1km_oninkb.toml",
        "convex_1km_oninkbr.toml",
        "flat_1km_lebreton.toml",
        "flat_1km_mheen.toml",
        "flat_1km_naive.toml",
        "flat_1km_oninkb.toml",
        "flat_1km_oninkbr.toml",
        "flat_2km_lebreton.toml",
        "flat_2km_mheen.toml",
        "flat_2km_naive.toml",
        "flat_2km_oninkb.toml",
        "flat_2km_oninkbr.toml",
        ]

    FORCE_RERUN = False # Defines whether or not to rerun simulations that have already been run
    pbar = tqdm(configs_to_run)

    for fname in pbar:
        cfg = CONFIG_FOLDER / fname
        cfg = toml.load(cfg)
        pbar.set_description(f"Running model ID {cfg['ID_human']} from {fname}")
        print(run_particles_from_cfg(cfg, force_rerun=FORCE_RERUN))


