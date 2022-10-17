from pathlib import Path
import matplotlib

# Setting matplotlib config
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    "axes.grid" : True,
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 8,
})

HYDRO_FOLDER = Path("sim-runs-hydro")
PARTICLE_SIM_FOLDER = Path("sim-runs-particles")
PARTICLE_PLOT_FOLDER = Path("visualisations_particles")
CONFIG_FOLDER = Path("sim-configs")
TEMPLATE_FOLDER = Path("templating")

# Create all folders if they don't exist
for folder in [HYDRO_FOLDER, PARTICLE_SIM_FOLDER, PARTICLE_PLOT_FOLDER, CONFIG_FOLDER, TEMPLATE_FOLDER]:
    if not folder.exists():
        folder.mkdir()

# Simulation options. Mappings from code to keyword used in sim config filenames
beaching_strats = {
    "A": "naive",
    "B": "lebreton",
    "C": "oninkb",
    "D": "mheen",
    "E": "oninkbr",
}
coast_shapes = {
    "X": "flat",
    "Y": "concave",
    "Z": "convex"
}
resolutions = {"1": "1km", "2": "2km"}

