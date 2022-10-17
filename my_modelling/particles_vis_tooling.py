from typing import Tuple

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import toml
import seaborn as sns

from env import PARTICLE_SIM_FOLDER, CONFIG_FOLDER

# ============================================================
# METHODS TO PROCESS PARTICLE DATA
# ============================================================
def clean_particle_file(ds: xr.Dataset) -> xr.Dataset:
    """
    Does some cleaning up of the particle file, and returns a new dataset.

    Added coords:
    - days_since_start: Additional coordinate variable related to OBS

    Other cleaning:
    - Drops the z variable (only 2D simulation)
    - Converts `in_beaching_region` and `beached` to boolean values
    - Drops the `traj` dimension for the `time` variable (particles all released at the same time) and assign as coord for `obs`
    - Drops the `obs` dimension for the `trajectory` variable (particles have same UID for all time) and assign as coord for `traj`
    """
    ds = ds.drop("z")

    # Converting dtypes
    ds = ds.assign(
        in_beaching_region = xr.where((0.0 < ds.land_value) & (ds.land_value < 1.0), x = True, y = False),
        beached = xr.where(ds.beached == 1.0, x = True, y = False),
    )

    # Collapsing dimensions for trajectory and time
    # Trajectory ID is defined for all particles at obs=0
    ds = ds.assign(
        trajectory = ds.trajectory.isel(obs = 0),
    )

    # Cleaning and assigning time coord
    all_times = (~np.isnat(ds.time)).all(dim = "obs") # Array of [traj] with True for particles where time is defined for all
    traj_index = np.argmax(all_times.values) # Finding index of first particle with time defined for all

    if traj_index == 0 and all_times.values[0] == False:
        # ie. all trajectories have incomplete times, can't create time coord
        raise Exception("All trajectories have incomplete times. Can't assign coordinate without filling in missing times.")

    ds = ds.assign(
        time = ds.time.isel(traj = traj_index),
    )

    # Creating `days_since_start` variable from np.datetime64 `time`
    days = ds.time.astype(float) / (1e9 * 60 * 60 * 24) # Converting from nanoseconds to days
    days -= days[0] # Setting first timestep of simulation to 0
    ds = ds.assign(
        days_since_start = days,
    )

    # Assigning variables as coords
    ds = ds.assign_coords(
        {"time": ds.time, "trajectory": ds.trajectory, "days_since_start": ds.days_since_start}
        )
    return ds

def beaching_feature_generation(ds: xr.Dataset) -> xr.Dataset:
    """
    Generates features relevant to the analysis from the particle data, and adds them to the dataset.
    

    Beaching features:
    - ever_beached[traj] -> bool: Whether the particle ever beached
    - ever_in_beaching_region[traj] -> bool: Whether the particle ever entered the beaching region
    - particles_never_in_beaching_region[] -> int: Number of particles that never entered the beaching region
    - beached_obs[traj] -> float: Gives the time observation of when the particle beached (float to allow nan)
    - beaching_region_obs[traj] -> float: Gives the time observation of when the particle first entered beaching region (float to allow nan)
    - beached_lon[traj] -> float: Gives the longitude of the particle when it beached
    - beached_lat[traj] -> float: Gives the latitude of the particle when it beached
    - entering_beaching_region[traj, obs] -> int: Whether the particle is entering (1) or leaving (-1) the beaching region. 0 for staying in same region

    Other features:
    - in_domain[traj, obs]: Within simulation domain
    - ever_left_domain[traj]: Whether the particle ever left the domain

    Assumptions:
    - Relies on `clean_particle_file` being run first.
    - Assumes that particles are not instantiated on land (otherwise will report that particles are beached on the second timestep)
    """
    
    # Adding in variables
    ds = ds.assign(
        ever_in_beaching_region = ds.in_beaching_region.any(dim="obs"),
        ever_beached = ds.beached.any(dim="obs"),
        in_domain = ~ds.lon.isnull(), # ie. no nan's in position
        entering_beaching_region = (
            ds.in_beaching_region.astype(int).diff("obs")
            .pad(obs=1, constant_values=0).isel(obs=slice(None, -1)) # Padding start of obs axis with 0's
        ), 
    )
    ds = ds.assign(particles_never_in_beaching_region = (1 - ds.ever_in_beaching_region).sum()) 


    beached_obs = ds.beached.argmax(dim = "obs")
    beached_obs = xr.where(beached_obs > 0, x=beached_obs, y=np.nan)
    ds = ds.assign(
        beached_obs = beached_obs, # Index in which beaching occurred, if any
    )

    beaching_region_obs = ds.in_beaching_region.astype(float).argmax(dim = "obs")
    beaching_region_obs = xr.where(beaching_region_obs > 0, x=beaching_region_obs, y=np.nan)
    ds = ds.assign(
        beaching_region_obs = beaching_region_obs, # Index in which the particle entered beaching region
    )

    beached_lon = xr.full_like(ds.beached_obs, fill_value = np.nan)
    beached_lat = beached_lon.copy()

    for traj_index, beached_obs_index in enumerate(ds.beached_obs.values):
        if np.isnan(beached_obs_index):
            continue
        beached_obs_index = int(beached_obs_index)
        beached_lon.loc[{"traj": traj_index}] = ds.lon.isel(traj = traj_index, obs = beached_obs_index)
        beached_lat.loc[{"traj": traj_index}] = ds.lat.isel(traj = traj_index, obs = beached_obs_index)
    
    ds = ds.assign(
        beached_lon = beached_lon,
        beached_lat = beached_lat,
        ever_left_domain = 1 - ds.in_domain.isel(obs = -1), # ie. if particle is in domain at last timestep, then it never left
    )

    return ds


def drop_particles(ds: xr.Dataset) -> xr.Dataset:
    """
    Drops particles that never entered the beaching region from the dataset, and returns a new dataset.

    Assumptions:
    - Relies on `beaching_feature_generation` being run first.
    """
    return ds.where(ds.ever_in_beaching_region == True, drop = True).load() # Loads in dataset

def particle_feature_generation(ds: xr.Dataset) -> xr.Dataset:
    """
    Generates particle features and adds them to the dataset

    Particle features:
    TODO: particle_speed[traj, obs]: Speed of the particle
    TODO: particle_velocity_x[traj, obs]: Speed of the particle in the x direction
    TODO: particle_velocity_y[traj, obs]: Speed of the particle in the y direction
    TODO: particle_distance_travelled[traj, obs]: Distance travelled by the particle
    """
    ...

class ParticlesSimulation:
    def __init__(self, cfg, all_particles):
        self.sim_code: str = cfg["codes"]["total"]
        self.toml_fname: str = cfg["toml_fname"]
        self.name = self.toml_fname.split(".")[0]

        self.cfg: dict = cfg
        self.run_id: str = cfg["particle_model"]["ID_human"]

        self.sim_folder = PARTICLE_SIM_FOLDER / self.run_id

        self.trajectories_fname = self.sim_folder / "trajectories.nc"
        self.fieldset_fname = self.sim_folder / "fieldset.nc"

        # Loading data
        self.all_particles = all_particles # Whether to load all particles into `ds_trajectories` or only the ones that enter beaching region

        ds = self.load_particles(self.trajectories_fname)
        if all_particles:
            self.ds_trajectories = ds
        else:
            self.ds_trajectories = drop_particles(ds)

        self.data = ParticleFileData(self.ds_trajectories)
        self.ds_fieldset = xr.open_dataset(self.fieldset_fname)
    
    @classmethod
    def from_toml(cls, toml_fname, all_particles = False):
        cfg = toml.load(CONFIG_FOLDER / toml_fname)
        return cls(cfg, all_particles)

    @staticmethod
    def load_particles(path):
        return (
            xr.open_dataset(path)
            .pipe(clean_particle_file)
            .pipe(beaching_feature_generation)
        )

class ParticleFileData:
    """
    A convenience class for accessing particle data for plotting.
    """
    def __init__(self, ds_trajectories):
        self._ds = ds_trajectories



    @property
    def beached_lon(self):
        """
        Gives the beached longitude for all particles that entered the beaching region
        Provides nans for particles the didn't beach.
        #! Only gives the first beaching location, not the last
        """
        return self.beached_info()[0]

    @property
    def beached_lat(self):
        """
        Gives the beached latitude for all particles that entered the beaching region
        Provides nans for particles the didn't beach
        """
        return self.beached_info()[1]

    @property
    def beached_days(self):
        """
        Gives the beached latitude for all particles that entered the beaching region
        Provides nans for particles the didn't beach
        """
        return self.beached_info()[2]

    @property
    def beached_lon_last(self):
        """
        Gives the final beached longitude for all beached particles
        Provides nans for particles the didn't beach.
        """
        ds = self._ds
        ds_last = ds.isel(obs = -1)
        return ds_last.where(ds_last.beached == 1).lon.values


    def beached_info(self):
        ds = self._ds

        # Plotting beaching locations
        beached_lon = ds.beached_lon
        beached_lat = ds.beached_lat
        beaching_days = np.full_like(beached_lon.values, np.nan) # Days after which the particle beached
        
        mask = ~np.isnan(beached_lon) # Which particles beach


        beaching_time_indices = ds.where(mask, drop=True).beached_obs.astype(int).values # Particles that actually beached
        beaching_days[mask.values] = ds.days_since_start.isel(obs = beaching_time_indices).values

        return beached_lon.values, beached_lat.values, beaching_days


# PLOTTING METHODS
def plot_particle_trajectories(ds: xr.Dataset, ax: plt.Axes = None, n_skip: int = None, xlim: tuple = None, ylim: tuple = None):
    """
    Plots the trajectories of all particles in the dataset.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    if n_skip is None:
        n_skip = 1
    
    for particle in range(0, len(ds.traj), n_skip):
        lons = ds.isel(traj = particle).lon
        lats = ds.isel(traj = particle).lat
        ax.plot(lons, lats)

    ax.set(
        xlim = xlim,
        ylim = ylim,
        aspect = "equal",
        xlabel = "Longitude [m]",
        ylabel = "Latitude [m]",
        title = "Particle Trajectories",
    )

    return fig, ax

# Function that returns matplotlib figure and axes objects
def plot_number_beached(sim_obj: ParticlesSimulation, ax: plt.Axes = None, label = None) -> plt.Axes:
    """
    Plot the raw number of particles that have been beached over time.
    """
    ds = sim_obj.ds_trajectories

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure


    beaching_time_indices = ds.beached_obs.where(~np.isnan(ds.beached_obs), drop=True).astype(int).values

    beaching_days = ds.days_since_start[beaching_time_indices].values
    beaching_days = np.sort(beaching_days)

    x = beaching_days # The day at which each particle beached
    y = np.arange(1, len(beaching_days)+1)

    # Plot with a step function
    ax.step(x, y, label = label)
    ax.set(
        xlabel="Time [days]",
        ylabel="Proportion beached",
        title="Number of particles beached",
    )
    ax.legend()
    return fig, ax

def plot_proportion_beached(sim_obj: ParticlesSimulation, ax: plt.Axes = None, label = None, beaching_region = True) -> plt.Axes:
    """
    Plot the proportion of particles that have been beached over time.

    This plotting function has two options:
    - beaching_region = False: looking at the proportion based on the total number of particles that beached (totals to 100% at the end)
    - beaching_region = True: looking at the proportion based on the particles that entered the beaching region.
    """
    ds = sim_obj.ds_trajectories

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure


    beaching_days = sim_obj.data.beached_days
    beaching_days = beaching_days[~np.isnan(beaching_days)]
    beaching_days = np.sort(beaching_days)

    x = beaching_days # The day at which each particle beached
    y = np.arange(1, len(beaching_days)+1)
    if beaching_region:
        title = "Particles beached / (total particles that entered the beaching region)"
        y = y / ds.ever_in_beaching_region.sum().item() * 100 # Normalising by the number of particles that entered the beaching region and converting to %
    else:
        title = "Proportion of particles beached"
        y = y / len(beaching_days) * 100 # Normalising by the total number of particles that beached and converting to %

    # Plot with a step function
    ax.step(x, y, label = label)
    ax.set(
        xlabel="Time [days]",
        ylabel="Proportion beached",
        title=title,
        ylim = [0, 105],
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc = "lower right")
    return fig, ax


def plot_beaching_density(sim_obj: ParticlesSimulation, ax = None, kde_kwargs = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the density plot over the x axis of the particles that have been beached.
    """
    ds_fieldset = sim_obj.ds_fieldset

    # Plotting beaching locations (leaving in nans for those that didn't beach but entered the region)
    lon = sim_obj.data.beached_lon
    lon = lon[~np.isnan(lon)] # ! Checking implementation is the same
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (6, 6))
    else:
        fig = ax.get_figure()
    # Plotting density plot of beaching longitudes
    if kde_kwargs is None:
        kde_kwargs = {}
    sns.kdeplot(np.array(lon), ax=ax, **kde_kwargs)
    ax.set(
        xlabel = "x [m]",
        ylabel = "Density per m",
        title = "Density of beaching locations",
        xlim = [ds_fieldset.x.min().item(), ds_fieldset.x.max().item()],
    )

    fig.tight_layout()
    return fig, ax

def plot_time_density(sim_obj: ParticlesSimulation, ax = None, kde_kwargs = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the density plot over the x axis of the particles that have been beached.
    """
    # Plotting beaching locations (leaving in nans for those that didn't beach but entered the region)
    days = sim_obj.data.beached_days
    days = days[~np.isnan(days)] # ! Checking implementation is the same
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (6, 6))
    else:
        fig = ax.get_figure()
    # Plotting density plot of beaching longitudes
    if kde_kwargs is None:
        kde_kwargs = {}
    sns.kdeplot(days, ax=ax, **kde_kwargs)
    ax.set(
        xlabel = "Days since start",
        ylabel = "Density per day",
        title = "Time density of beaching",
    )

    fig.tight_layout()
    return fig, ax





def summarise_trajectories(sim_obj: ParticlesSimulation):
    """
    Returns a dictionary of summary statistics for the simulations.
    """
    ds = sim_obj.ds_trajectories
    entered_beaching_region = ds.ever_in_beaching_region.sum().item()
    summary = {}
    summary["Simulation code"] = sim_obj.cfg["codes"]["total"]
    summary["Initial particles"] = len(ds.traj) # Initial no. of particles
    summary["Particles in beaching region"] = int(entered_beaching_region) # No. of particles entered beaching region
    summary["Proportion in beaching region"] = entered_beaching_region/sim_obj.cfg["particle_model"]["n_particles"] # No. of particles entered beaching region
    summary["Particles exiting domain while having entered beaching region"] = int((ds.ever_in_beaching_region == ds.ever_left_domain).sum().item()) # No. of particles exiting domain while having entered beaching region

    summary["Particles beached"] = int(ds.isel(obs=-1).beached.sum().item()) # No. of particles beached
    summary["Proportion beached"] = ds.isel(obs=-1).beached.sum().item() / entered_beaching_region # proportion of particles beached that entered beaching region
    summary["Crossings into beaching region"] = int(ds.where(ds.entering_beaching_region == 1).entering_beaching_region.sum().item()) # No. of crossings into beaching region
    summary["Crossings out of beaching region"] = int(np.abs(ds.where(ds.entering_beaching_region == -1).entering_beaching_region.sum().item())) # No. of crossings out of beaching region
    return summary

def process_landpoints_to_cells(array: np.ndarray) -> np.ndarray:
    """
    Takes in the 2D land mask of the fieldset (which indicates where land and ocean are for points)
    and returns a corresponding array for cells of dim: shape - 1

    Used for plotting the mask of the land.
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    array = array.astype(dtype=bool) # True for gridpoints that are land
    # Create a new array true for all cells completely surrounded by land gridpoints
    top_left = array[:-1, :-1]
    top_right = array[:-1, 1:]
    bottom_left = array[1:, :-1]
    bottom_right = array[1:, 1:]
    return top_left & top_right & bottom_left & bottom_right
