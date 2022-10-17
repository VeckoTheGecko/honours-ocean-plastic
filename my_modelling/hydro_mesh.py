import typing
import os
import numpy as np
import xarray as xr
import delft3d


from matplotlib.image import imread

from pathlib import Path


class HydroMesh:
    """
    A class to aid in the creationg of the hydrodynamic grid from images or numpy arrays.
    This method is capable of both exporting to the .grd file to be used in Delft3D, as well as the .nc file to be used as a land mask within
    parcels.
    """
    def __init__(
        self,
        x_start: float,
        y_start: float,
        x_stop: float = None,
        y_stop: float = None,
        dx: float = None,
        dy: float = None,
        ) -> None:
        # Validating the inputs to the constructor
        if sum([x_stop is not None, dx is not None]) != 1:
            raise ValueError(f"Only one of dx or x_stop must be None. dx={dx}, x_stop={x_stop}")

        if sum([y_stop is not None, dy is not None]) != 1:
            raise ValueError(f"Only one of dy or y_stop must be None. dy={dy}, y_stop={y_stop}")

        self.x_start = x_start
        self.y_start = y_start
        self.x_stop = x_stop
        self.y_stop = y_stop
        self.dx = dx
        self.dy = dy

        # Instantiating variables to be set later
        self.x: np.ndarray = None
        self.y: np.ndarray = None
        self.X: np.ndarray = None
        self.Y: np.ndarray = None
        self.bool_cells: np.ndarray = None # Array of true for ocean cells
        self.bool_mesh: np.ndarray = None # Array of true for ocean gridpoints
        self.grd_file: delft3d.GrdFile # Delft gridfile

    def from_png(self, path: typing.Union[str, bytes, os.PathLike]) -> None:
        """
        Reads a PNG image stores it as mesh data in the HydroMesh object.
        """
        rgb_array = imread(path)
        rgb_array = rgb_array[:, :, :3] # Trimming off A of RGBA if provided
        rgb_array = rgb_array[::-1, :, :] # Reversing rows: (0, 0) is top-left for images, making it bottom left
        return self.from_image(rgb_array)
        
    def from_image(self, array: np.ndarray):
        """
        Takes in array of rgb values (shape: rows, cols) and converts to (rows, cols)
        boolean array with False values for white (RGB=(1, 1, 1)) pixels.

        Note: Colors need to be in the range 0-1.
        """
        # TODO: Smarter handling of RGBA alphas being non-defined?
        
        # Create array of Trues for white pixels
        bools = np.logical_and(
            *(array[:, :, i] == 1 for i in range(3))
            )
        bools = ~bools # Becomes "True" for ocean cells (non-white)
        return self.from_array(bools)
    
    def from_array(self, array: np.ndarray) -> None:
        """
        Reads a numpy array of boolean values and stores it as mesh data in the HydroMesh object.
        """
        # Number of cells
        n_x = array.shape[1]
        n_y = array.shape[0]
        
        # Calculating x_stop, y_stop, dx, dy if not provided
        if self.x_stop is None:
            self.x_stop = self.x_start + n_x * self.dx
        elif self.dx is None:
            self.dx = (self.x_stop - self.x_start) / n_x

        if self.y_stop is None:
            self.y_stop = self.y_start + n_y * self.dy
        elif self.dy is None:
            self.dy = (self.y_stop - self.y_start) / n_y
            
        
        x = np.linspace(self.x_start, self.x_stop, n_x + 1, endpoint = True)
        y = np.linspace(self.y_start, self.y_stop, n_y + 1, endpoint = True)

        X, Y = np.meshgrid(x, y)

        # Convert booleans for cells to booleans for mesh points
        self.bool_cells = array
        bool_mesh = to_bool_mesh(self.bool_cells)
        assert bool_mesh.shape == X.shape, f"bool_mesh.shape={bool_mesh.shape}, X.shape={X.shape}. Bool mesh shape and X shape must be equal"

        # Storing variables to be used in the exporting of the mesh to .grd and .nc files
        self.x = x # x-coordinates of mesh points (1D)
        self.y = y # y-coordinates of mesh points (1D)
        self.X = X # ... (2D)
        self.Y = Y # ... (2D)
        self.bool_mesh = bool_mesh # Boolean array indicating land for meshpoints (2D)
        return

    def to_grd(
        self,
        template_grd:typing.Union[str, bytes, os.PathLike],
        out_grd: typing.Union[str, bytes, os.PathLike],
        missing_value: float = -999.0
        ) -> None:
        """
        Exports the mesh to a .grd file to be used in Delft3D.
        """
        grd = delft3d.GrdFile(template_grd)
        print(grd.header)

        X = self.X
        Y = self.Y
        X[~self.bool_mesh] = missing_value
        Y[~self.bool_mesh] = missing_value

        grd.set_gird(X, Y, "Cartesian")

        grd.to_file(out_grd)
        self.grd_file = grd
        return
        

    def to_netcdf(self, path: typing.Union[str, bytes, os.PathLike], land_value: float = 1.0, sea_value: float = 0.0) -> None:
        """
        Exports the mesh to a .nc file to be used as a land mask in parcels.

        Parameters
        ----------
        path : str
            Path to the .nc file to be created.
        land_value : float
            Value to be assigned to land pixels.
        sea_value : float
            Value to be assigned to sea pixels.
        """
        ds = self.to_dataset(land_value, sea_value)
        ds.to_netcdf(path)
        return        

    def to_dataset(self, land_value: float = 1.0, sea_value: float = 0.0) -> None:
        """
        Converts the mesh to a dataset containing a land mask to be used in parcels.

        This land mask needs to be manually expanded by one gridcell. In delft, the edge of the ocean mesh is the interface with the land.
        The edge is shared, hence taking the opposite of this mesh to define as the land would be innacurate (and could contaminate the whole set
        of simulations).) 

        Parameters
        ----------
        land_value : float
            Value to be assigned to land pixels.
        sea_value : float
            Value to be assigned to sea pixels.
        """
        # Converting to float if not already
        land_value, sea_value = float(land_value), float(sea_value)

        # Constructing land array
        is_land = to_bool_mesh(~self.bool_cells)
        land = np.empty_like(is_land, dtype=np.float32)
        land[is_land] = land_value
        land[~is_land] = sea_value

        data_vars = {
            "lon_mesh": (("lat", "lon"), self.X),
            "lat_mesh": (("lat", "lon"), self.Y),
            "land": (("lat", "lon"), land),
        }
        coords = {
            "lon": self.x,
            "lat": self.y,
        }

        ds = xr.Dataset(
            data_vars,
            coords,
        )
        return ds


def to_bool_mesh(bool_cells: np.ndarray) -> np.ndarray:
    """Takes in boolean array of cells and converts to a boolean array of mesh points.

    Implementation: If a cell is True, all corresponding points at the cells vertices are also True.

    WARNING: Inherently bool cells and bool mesh points are not equivalent.
    Some bool cells configurations are impossible to convert to bool mesh (eg. one
    pixel holes). This implementation is good enough for these purposes.

    :param bool_cells: Array of booleans indicating True/False for a
    :type bool_cells: np.array
    :return: Array of booleans indicating True/False for individual points on a mesh.
    :rtype: np.array
    """
    # Create arrays mapping a cell to each of the corners
    top_left = np.pad(bool_cells, ((0, 1), (0, 1)), 'constant', constant_values=False)
    top_right = np.pad(bool_cells, ((0, 1), (1, 0)), 'constant', constant_values=False)
    bottom_left = np.pad(bool_cells, ((1, 0), (0, 1)), 'constant', constant_values=False)
    bottom_right = np.pad(bool_cells, ((1, 0), (1, 0)), 'constant', constant_values=False)
    
    # Combine all four corners into a single array.
    # If any of the above are True, return True for the corresponding meshpoint.
    return top_left | top_right | bottom_left | bottom_right

