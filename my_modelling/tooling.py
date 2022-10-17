"""
A variety of tooling/utility functions for use in a variety of places in the codebase/modelling pipeline.
"""
import numpy as np


def von_neuman_single(bool_array: np.ndarray):
    """
    Converts False cells in the von Neuman neighbourhood of True cells to True.
    """
    north = np.pad(bool_array[1:, :], ((0, 1), (0, 0)), 'constant', constant_values=False)
    south = np.pad(bool_array[:-1, :], ((1, 0), (0, 0)), 'constant', constant_values=False)
    east = np.pad(bool_array[:, 1:], ((0, 0), (0, 1)), 'constant', constant_values=False)
    west = np.pad(bool_array[:, :-1], ((0, 0), (1, 0)), 'constant', constant_values=False)
    return bool_array | north | south | east | west


def moore_single(bool_array: np.ndarray):
    """
    Converts False cells in the Moore neighbourhood of True cells to True.
    """
    north = np.pad(bool_array[1:, :], ((0, 1), (0, 0)), 'constant', constant_values=False)
    south = np.pad(bool_array[:-1, :], ((1, 0), (0, 0)), 'constant', constant_values=False)
    east = np.pad(bool_array[:, 1:], ((0, 0), (0, 1)), 'constant', constant_values=False)
    west = np.pad(bool_array[:, :-1], ((0, 0), (1, 0)), 'constant', constant_values=False)
    north_east = np.pad(bool_array[1:, 1:], ((0, 1), (0, 1)), 'constant', constant_values=False)
    north_west = np.pad(bool_array[1:, :-1], ((0, 1), (1, 0)), 'constant', constant_values=False)
    south_east = np.pad(bool_array[:-1, 1:], ((1, 0), (0, 1)), 'constant', constant_values=False)
    south_west = np.pad(bool_array[:-1, :-1], ((1, 0), (1, 0)), 'constant', constant_values=False)
    return bool_array | north | south | east | west | north_east | north_west | south_east | south_west


def von_neuman(bool_array: np.ndarray, n: int):
    """
    Converts False cells, that are von Neuman neighbours distance n or less from a True cell, to True.
    """
    for i in range(n):
        bool_array = von_neuman_single(bool_array)
    return bool_array

def moore(bool_array: np.ndarray, n: int):
    """
    Converts False cells, that are Moore neighbours distance n or less from a True cell, to True.
    """
    for i in range(n):
        bool_array = moore_single(bool_array)
    return bool_array

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 3x3 matrix of booleans with a single True in the middle
    bools = np.full(shape = (11,11), fill_value = False)
    bools[5,5] = True

    fig, axs = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axs.flatten()):
        ax.imshow(von_neuman(bools, i), vmin = 0, vmax = 1)
        ax.set_title(f"n = {i}")
    plt.show()


def set_attr(obj, path, value):
    # Adapted from https://stackoverflow.com/a/69572347/15545258
    return_dict = obj
    *path, last = path.split(".")
    for bit in path:
        obj = obj.setdefault(bit, {})
    obj[last] = value
    return return_dict
