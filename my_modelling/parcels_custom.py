from typing import Callable
import parcels.rng as ParcelsRandom
from parcels.application_kernels.advection import AdvectionRK4
from parcels import JITParticle, BaseParticleSet, Variable, ErrorCode, FieldSet, ParticleSet
import math
import numpy as np
import functools
# I break PEP8 naming conventions in this file. Sorry mum and dad :-(

class BeachingStrategy:
    """
    This class packages together particle classes, fieldset annotating methods, and beaching kernels which are to be used for the beaching strategies.

    - Particle classes:
        - Contains particle specific information and variables that are required for the beaching strategies to function.
    - Fieldset annotating methods:
        - Methods that add variables to the fieldset that are required for the beaching strategies to function.
    - Beaching kernels:
        - Kernel combination used to determine the movement and beaching behaviour for the particles
    """
    def __init__(self, particle: JITParticle, kernel_lst: list[Callable], fieldset_transform: Callable | None):
        self._particle = particle
        self._kernel_lst = kernel_lst
        self._fieldset_transform = fieldset_transform
    
    def get_fieldset(self, fieldset, **kwargs):
        """Updates the fieldset using the fieldset options provided in the configuration dictionary.
        """
        if self._fieldset_transform is None:
            # Check if dict is empty
            if kwargs:
                raise ValueError(f"Beaching strategy does not require fieldset options, but options were provided: {kwargs}")
            return fieldset
        return self._fieldset_transform(fieldset, **kwargs)
    
    def get_kernel(self, pset: BaseParticleSet):
        """
        Returns the kernel combination for the beaching strategy.
        """
        kernels = (pset.Kernel(i) for i in self._kernel_lst)
        return functools.reduce(lambda x, y: x+y, kernels)
    
    def get_particle(self):
        """Returns the particle class for the beaching strategy.
        """
        return self._particle
    
    @property
    def recovery(self):
        return {ErrorCode.ErrorOutOfBounds: DeleteParticle}


# KERNELS
# Common kernels
def DiffusionUniformKh_Beaching(particle, fieldset, time):
    # Code adapted from AdvectionRK4 https://oceanparcels.org/gh-pages/html/_modules/parcels/application_kernels/advection.html#AdvectionRK4
    # Wiener increment with zero mean and std of sqrt(dt)
    if particle.beached == 0.0:
        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])

        particle.lon += bx * dWx
        particle.lat += by * dWy


def AdvectionRK4_Beaching(particle, fieldset, time):
    # Code adapted from AdvectionRK4 https://oceanparcels.org/gh-pages/html/_modules/parcels/application_kernels/advection.html#AdvectionRK4
    if particle.beached == 0.0:
        (u1, v1) = fieldset.UV[particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt


def DeleteParticle(particle, fieldset, time):
    particle.delete()



# Beaching kernels
def OninkBeachingKernel(particle, fieldset, time):
    """
    Beaches the particle if it is within one gridcell of the land.

    """
    particle.land_value = fieldset.land[particle] # ie. if land = 0, in ocean. If 0 < land < 1, in beaching region. If land = 1, on land (collided with coast).
    proba_beach = fieldset.beaching_constant
    proba_dt = 1.0 - math.exp(- particle.dt / proba_beach) # Converting to probability to apply for each timestep
    if particle.beached == 0.0 and fieldset.land[particle] > 0.0 and ParcelsRandom.random() < proba_dt: # ie. particle is floating and is in beaching region
        particle.beached = 1.0

def OninkResusKernel(particle, fieldset, time):
    """
    Beaches the particle if it is within one gridcell of the land.

    """
    proba_resus = fieldset.resuspension_constant
    proba_dt = 1.0 - math.exp(- particle.dt / proba_resus) # Converting to probability to apply for each timestep
    if particle.beached == 1.0 and ParcelsRandom.random() < proba_dt: # ie. particle is floating and is in beaching region
        particle.beached = 0.0


def MheenBeachingKernel(particle, fieldset, time):
    """
    Beaches the particle with probability if it is within one gridcell of the land and moving towards.
    """
    particle.land_value = fieldset.land[particle] # ie. if land = 0, in ocean. If 0 < land < 1, in beaching region. If land = 1, on land (collided with coast).
    proba_beach = fieldset.beaching_constant
    proba_dt = 1.0 - math.exp(- particle.dt / proba_beach) # Converting to probability to apply for each timestep
    if particle.beached == 0.0 and particle.land_value > particle.previous_land_value and fieldset.land[particle] > 0.0 and ParcelsRandom.random() < proba_dt: # ie. particle is floating and moving towards land
        particle.beached = 1.0
    
    particle.previous_land_value = particle.land_value


def NaiveBeachingKernel(particle, fieldset, time):
    """
    Beaches the particle naively (ie. if the particle collides with the land and stops moving).

    """
    particle.land_value = fieldset.land[particle] # ie. if land = 0, in ocean. If 0 < land < 1, in beaching region. If land = 1, on land (collided with coast).
    if fieldset.land[particle] > 0.0: # ie. the particle has hits the coastline (speed is 1% of the average)
        particle.beached = 1.0


def LebretonBeachingKernel(particle, fieldset, time):
    """
    Beaches the particle if it has spent more than x days within one gridcell of the land.

    """
    particle.land_value = fieldset.land[particle] # ie. if land = 0, in ocean. If 0 < land < 1, in beaching region. If land = 1, on land (collided with coast).
    if fieldset.land[particle] > 0.0: # ie. particle is floating
        particle.time_in_beaching_region += particle.dt
    if fieldset.land[particle] == 0.0:
        particle.time_in_beaching_region = 0.0


    if particle.time_in_beaching_region > fieldset.beaching_time_cutoff:
        particle.beached = 1.0

def KnockbackKernel(particle, fieldset, time):
    """
    If a particle directly hits land, the particle gets moved back to its previous position
    """
    if fieldset.land[particle] == 1.0:
        particle.lon = particle.previous_lon
        particle.lat = particle.previous_lat

    particle.previous_lon = particle.lon
    particle.previous_lat = particle.lat

def BorderKernel(particle, fieldset, time):
    """
    If a particle directly is close to land, it gets a nudge out to the ocean
    """
    if fieldset.land[particle] > 0.9:
        right = math.floor(fieldset.land[time, particle.depth, particle.lat, particle.lon + 1_000])
        left = math.floor(fieldset.land[time, particle.depth, particle.lat, particle.lon - 1_000])
        up = math.floor(fieldset.land[time, particle.depth, particle.lat + 1_000, particle.lon])
        down = math.floor(fieldset.land[time, particle.depth, particle.lat - 1_000, particle.lon])
        x = -(right - left) # Move opposite to direction of land
        y = -(up - down)
        x = x / math.sqrt(x**2 + y**2)
        y = y / math.sqrt(x**2 + y**2)
        particle.lon = particle.lon + x * particle.dt # 1m/s nudge for dt out to sea
        particle.lat = particle.lat + y * particle.dt

# PARTICLES
class BeachingParticle(JITParticle):
    #Now the beaching variables
    #0=open ocean, 1=beached
    beached = Variable(
        'beached', dtype=np.int32,
        initial=0,
    )
    # Land value at the particle's location
    # 0 -> ocean
    # 1 -> land
    # 0 < land_value < 1 -> beaching_region
    land_value = Variable(
        'land_value', dtype=np.float32,
        initial=0.0,
    )

    previous_lon = Variable(
        'previous_lon', dtype=np.float32,
        initial=0.0,
        to_write=False,
    )
    previous_lat = Variable(
        'previous_lat', dtype=np.float32,
        initial=0.0,
        to_write=False,
    )

OninkBeachingParticle = BeachingParticle


class MheenBeachingParticle(BeachingParticle):
    # Land value at the particle's location
    # 0 -> ocean
    # 1 -> land
    # 0 < land_value < 1 -> beaching_region
    previous_land_value = Variable(
        'previous_land_value', dtype=np.float32,
        initial=0.0,
        to_write=False,
    )

class LebretonBeachingParticle(BeachingParticle):
    # Land value at the particle's location
    # 0 -> ocean
    # 1 -> land
    # 0 < land_value < 1 -> beaching_region
    time_in_beaching_region = Variable(
        'time_in_beaching_region', dtype=np.int32,
        initial=0,
    )



# FIELDSET TRANSFORMATIONS
def add_beaching_timescale(fieldset, beaching_timescale_days):
    seconds = beaching_timescale_days * (24 * 60**2)
    fieldset.add_constant('beaching_constant', seconds)
    return fieldset


def add_timescales(fieldset, beaching_timescale_days, resuspension_timescale_days):
    seconds = beaching_timescale_days * (24 * 60**2)
    fieldset.add_constant('beaching_constant', seconds)
    seconds = resuspension_timescale_days * (24 * 60**2)
    fieldset.add_constant('resuspension_constant', seconds)
    return fieldset

def add_beaching_time_cutoff(fieldset, beaching_time_cutoff_days):
    """
    Specifies the number of days a particle needs to be in the beaching region before it is beached.

    Facilitates Lebretons beaching kernel.
    """
    seconds = beaching_time_cutoff_days * (24 * 60**2)
    fieldset.add_constant('beaching_time_cutoff', seconds)
    return fieldset


# The following dictionary specifies the combinations
# TODO: Specify kernels
beaching_mappings = {
    "Mheen2020" : BeachingStrategy(
        particle = MheenBeachingParticle,
        fieldset_transform = add_beaching_timescale,
        kernel_lst = [
            AdvectionRK4_Beaching,
            DiffusionUniformKh_Beaching,
            MheenBeachingKernel,
            BorderKernel,
        ]
    ),
    "Onink2021-beach": BeachingStrategy(
        particle = OninkBeachingParticle,
        fieldset_transform = add_beaching_timescale,
        kernel_lst = [
            AdvectionRK4_Beaching,
            DiffusionUniformKh_Beaching,
            OninkBeachingKernel,
            BorderKernel,
        ]
        ),
    "Onink2021-beach-resus": BeachingStrategy(
        particle = OninkBeachingParticle,
        fieldset_transform = add_timescales,
        kernel_lst = [
            AdvectionRK4_Beaching,
            DiffusionUniformKh_Beaching,
            OninkBeachingKernel,
            OninkResusKernel,
            BorderKernel,
        ]
        ),
    "Lebreton2018": BeachingStrategy(
        particle = LebretonBeachingParticle,
        fieldset_transform = add_beaching_time_cutoff,
        kernel_lst = [
            AdvectionRK4_Beaching,
            DiffusionUniformKh_Beaching,
            LebretonBeachingKernel,
            BorderKernel,
        ],
    ),
    "naive": BeachingStrategy(
        particle = BeachingParticle,
        fieldset_transform = None,
        kernel_lst = [
            AdvectionRK4_Beaching,
            DiffusionUniformKh_Beaching,
            NaiveBeachingKernel,
            BorderKernel,
        ],
    ),
}

