# %% [markdown]
# # Config creator
# Creates config files for all the simulations wanting to be run. These files act as the backbone of the simulations, and act as definition files to run a simulation.
# 
# The config files also assign IDs for:
# - Coastlines
# - Hydro runs
# - Particle runs
# in order to uniquely identify model runs, and for effective caching.

# %%
import toml
from templating.utils import Hasher, dict_has_nones, remove_ids
from copy import deepcopy
from __init__ import logger
from pathlib import Path
from matplotlib.image import imread
from env import beaching_strats, coast_shapes, resolutions

# Baseline config. IDs are assigned by the generate_ids() function.
cfg_baseline = {
    # "ID": None,
    # "ID_human": None,
    "coastline": {
        # "ID": None,
        "coast_file": None,
        "start_xy": [0, 0],
        "dx": 1_000,
        "dy": 1_000,
    },
    "hydro_model": {
        # "ID": None,
        # "ID_human": None,
        "duration_days": 4,
    },
    "particle_model": {
        # "ID": None,
        # "ID_human": None,
        "duration_days": 30,
        "n_particles": 5_000,
        "dt_mins": 5,
        "K_h": 1,
        "particle_seed": 16, # Seed for the random number generator
        "spawning_band": 60_000, # particle spawning band in metres around land
        "beaching": {
            "beaching_key": None,
            "fieldset_options": {},
        }
    }
}

def generate_ids(settings: dict):
    """
    Generates IDs for the `settings` dictionary, which is used as the master settings file.
    
    Resulting structure:
    {
        "ID": "sha256 hash of the dictionary",
        "ID_human": "human readable hash of the dictionary",
        "coastline": {
            "ID": "sha256 hash of the dictionary",
            "ID_human": "human readable hash of the dictionary",
            ...
        },
        "hydro_model": {
            "ID": "sha256 hash of the dictionary",
            "ID_human": "human readable hash of the dictionary",
            ...
        },
        "particle_model": {
            "ID": "sha256 hash of the dictionary",
            "ID_human": "human readable hash of the dictionary",
            ...
        }
    }
    """
    if dict_has_nones(settings):
        logger.error("The settings config contains None values. Please fill in all values before generating IDs.")
        raise ValueError("The settings config contains None values. Please fill in all values before generating IDs.")

    logger.info("Generating IDs for the master settings file.")
    settings = remove_ids(settings) # Strip all IDs out if they're there
    hash_values = deepcopy(settings) # A helper dictionary which is used to store values used to compute the hash

    # Generate IDs for coastline
    logger.info(f"Generating IDs for the coastline using dict: {hash_values['coastline']}")
    if not Path(hash_values["coastline"]["coast_file"]).exists():
        raise FileNotFoundError(f"Coastline file {hash_values['coastline']['coast_file']} does not exist.")
    hasher = Hasher(hash_values["coastline"])
    settings["coastline"]["ID"] = hasher.hash
    settings["coastline"]["ID_human"] = hasher.human_hash
    # No need for human hash (not used in cache files etc.)

    # Generate IDs for hydro model
    # Depends on the coastline hash as well
    hash_values["hydro_model"]["coastline_ID"] = settings["coastline"]["ID"]
    logger.info(f"Generating IDs for the hydro_model using dict: {hash_values['hydro_model']}")
    hasher = Hasher(hash_values["hydro_model"])
    settings["hydro_model"]["ID"] = hasher.hash
    settings["hydro_model"]["ID_human"] = hasher.human_hash

    # Generate IDs for particle model
    # Depends on the hydro model hash as well
    hash_values["particle_model"]["hydro_ID"] = settings["hydro_model"]["ID"]
    logger.info(f"Generating IDs for the particle_model using dict: {hash_values['particle_model']}")
    hasher = Hasher(hash_values["particle_model"])
    settings["particle_model"]["ID"] = hasher.hash
    settings["particle_model"]["ID_human"] = hasher.human_hash

    # Total hash is same as particle hash
    settings["ID"] = settings["particle_model"]["ID"]
    settings["ID_human"] = settings["particle_model"]["ID_human"]

    logger.success(f"IDs generated. Final settings files is: {settings}")
    return settings


def validate_beaching_key(cfg: dict):
    """
    Validate and set the beaching key in the config file (so there are no errors down the line).
    """
    beaching_key = cfg["particle_model"]["beaching"]["beaching_key"]
    assert beaching_key is not None, "Beaching key is None. Please set a beaching key."

    from parcels_custom import beaching_mappings
    if beaching_key not in beaching_mappings.keys():
        raise ValueError(f"Beaching key {beaching_key} not valid.")
    return cfg

def get_codes(toml_name):
    """
    Get the code for the simulation from the name of the toml file.
    # ! Not the most robust way of doing this, but it works for now.
    """
    # Creating local versions of the dictionaries to pop items from (and avoid "referenced before assignment errors")
    beaching_strats_loc = beaching_strats.copy()
    coast_shapes_loc = coast_shapes.copy()
    resolutions_loc = resolutions.copy()

    # Removing the strategies that don't match
    items = list(beaching_strats_loc.items())
    for code, keyword in items:
        if keyword not in toml_name:
            beaching_strats_loc.pop(code)

    items = list(coast_shapes_loc.items())
    for code, keyword in items:
        if keyword not in toml_name:
            coast_shapes_loc.pop(code)

    items = list(resolutions_loc.items())
    for code, keyword in items:
        if keyword not in toml_name:
            resolutions_loc.pop(code)

    # Taking care of edgecase
    if all((strat in beaching_strats_loc.values() for strat in ["oninkb", "oninkbr"])):
        beaching_strats_loc = {key:value for key, value in beaching_strats_loc.items() if value != "oninkb"}
    
    if any((len(dic) != 1 for dic in [beaching_strats_loc, coast_shapes_loc, resolutions_loc])):
        raise ValueError(f"One of the codes didn't have a unique match. Matches for {toml_name} were {beaching_strats_loc, coast_shapes_loc, resolutions_loc}")
    
    return (unpack_item(coast_shapes_loc), unpack_item(resolutions_loc), unpack_item(beaching_strats_loc))


def unpack_item(dic):
    """
    Returns the key for a dict of length 1
    """
    return list(dic.keys())[0]