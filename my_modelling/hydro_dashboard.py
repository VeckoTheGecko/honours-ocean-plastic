from typing import Tuple
import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import altair as alt
from pathlib import Path
import shutil

from tqdm import tqdm

st.title("🌊 Hydro Dashboard")

# @st.cache(hash_funcs = {xr.core.dataset.Dataset: lambda ds: ds.values})
def load_data(path):
    from delft_to_parcels_tooling import delft_to_fieldset
    return delft_to_fieldset(path, return_ds=True)

def dt_to_string(dt: np.datetime64):
    return dt.astype('datetime64[us]').item().strftime('%Y-%m-%d %H:%M:%S') # As per https://stackoverflow.com/questions/28327101/cant-call-strftime-on-numpy-datetime64-no-definition

def print_stats(ds_grid: xr.Dataset) -> None:

    st.write("### Time")
    col1, col2 = st.columns(2)
    col1.metric("Simulation start", dt_to_string(ds_grid.time.values[0]))
    col1.metric("Simulation end", dt_to_string(ds_grid.time.values[-1]))

    col2.metric("Simulation duration", f"{(ds_grid.time.values[-1] - ds_grid.time.values[0]) / np.timedelta64(1, 'D') :.2f} days")
    col2.metric("Simulation dt of mesh data", f"{(ds_grid.time.values[1]-ds_grid.time.values[0]) / np.timedelta64(1, 'm') :.2f} mins")
    st.metric("Timesteps", f"{ds_grid.time.size}")
    st.write("---")
    st.write("### Space")
    col1, col2 = st.columns(2)
    col1.metric("x domain", f"[{ds_grid.x.min().item():.0f}, {ds_grid.x.max().item():.0f}]m")
    col1.metric("y domain", f"[{ds_grid.y.min().item():.0f}, {ds_grid.y.max().item():.0f}]m")
    col2.metric("dx step", f"{ds_grid.x.values[1]-ds_grid.x.values[0]} m")
    col2.metric("dy step", f"{ds_grid.y.values[1]-ds_grid.y.values[0]} m")
    st.metric("Gridsize", f"{ds_grid.x.size} x {ds_grid.y.size}")

    return

def get_figure(
    ds: xr.Dataset,
    time_index: int,
    ref_x_index: int,
    ref_y_index: int,
    c_ranges: dict[Tuple]
    ):
    """
    Get quadmeshes + linechart for the dataset.
    """
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))

    # AXES 0, 0
    # ===================
    ax = axs[0, 0]
    cmin, cmax = c_ranges["U"]
    quadmesh = ds.U.isel(time=time_index).plot(x = "x", y = "y", ax = ax, cmap = "RdBu_r")
    quadmesh.set_clim(vmin=cmin, vmax=cmax) # Setting colour limits

    # Plotting the station position
    ax.plot(ds.x.sel(x_index = ref_x_index), ds.y.sel(y_index = ref_y_index), marker="o", color="red", markersize=5)

    ax.set(
        title = f"U for timestep {time_index}"
        )

    # AXES 0, 1
    # ===================
    ax = axs[0, 1]
    cmin, cmax = c_ranges["V"]
    quadmesh = ds.V.isel(time=time_index).plot(x = "x", y = "y", ax = ax, cmap = "RdBu_r")
    quadmesh.set_clim(vmin=cmin, vmax=cmax) # Setting colour limits

    # Plotting the station position
    ax.plot(ds.x.sel(x_index = ref_x_index), ds.y.sel(y_index = ref_y_index), marker="o", color="red", markersize=5)
    ax.set(
        title = f"V for timestep {time_index}"
        )

    # AXES 0, 2
    # ===================
    ax = axs[0, 2]
    cmin, cmax = c_ranges["S"]
    quadmesh = ds.S.isel(time=time_index).plot(x = "x", y = "y", ax = ax, cmap = "RdBu_r")
    quadmesh.set_clim(vmin=cmin, vmax=cmax) # Setting colour limits

    # Plotting the station position
    ax.plot(ds.x.sel(x_index = ref_x_index), ds.y.sel(y_index = ref_y_index), marker="o", color="red", markersize=5)
    ax.set(
        title = f"S for timestep {time_index}"
        )
    
    # AXES 1, 0
    # ===================
    ax = axs[1, 0]
    # Plotting U timeseries
    U = ds.U.sel(x_index = ref_x_index, y_index = ref_y_index)
    U.plot(ax = ax, label = "U")
    # u_min_max = [ds_line.min().item(), ds_line.max().item()]

    # Pltoing V timeseries
    V = ds.V.sel(x_index = ref_x_index, y_index = ref_y_index)
    V.plot(ax = ax, label = "V")
    # v_min_max = [ds_line.min().item(), ds_line.max().item()]

    speed = np.sqrt(U.values**2 + V.values**2)
    ax.plot(ds.time, speed, label="Current speed")
    ax.axvline(U.isel(time=time_index).time.values, color="red", linestyle="--", linewidth = 1) # Plotting the time of the timestep

    ax.set(
        title = f"Velocity and speed timeseries",
        xlabel = "Time",
        ylabel = "Velocity or speed [m/s]"
    )
    ax.legend()

    # AXES 1, 1
    # ===================
    ax = axs[1, 1]
    ds_plot = ds.isel(time=time_index)
    ds_plot = ds_plot.assign(
        speed = np.sqrt(ds_plot.U**2 + ds_plot.V**2)
    )

    array = ds_plot.speed.values.flatten()
    counts, bins = np.histogram(array[~np.isnan(array)], bins = 200)
    ax.hist(bins[:-1], bins, weights=counts)
    ax.set(
        title = f"Speed distribution for timestep {time_index}",
        xlabel = "Speed (m/s)",
        ylabel = "Count"
    )

    # AXES 1, 2
    # ===================
    ax = axs[1, 2]
    # U velocity distribution
    array = ds_plot.U.values.flatten()
    counts, bins = np.histogram(array[~np.isnan(array)], bins = 200)
    ax.hist(bins[:-1], bins, weights=counts, label = "U")
    # V velocity distribution
    array = ds_plot.V.values.flatten()
    counts, bins = np.histogram(array[~np.isnan(array)], bins = 200)
    ax.hist(bins[:-1], bins, weights=counts, label = "V")
    ax.set(
        title = f"Velocity distribution for timestep {time_index}",
        xlabel = "Speed (m/s)",
        ylabel = "Count"
    )
    ax.legend()

    # Quiver plots
    for ax in axs.flatten()[:3]:
        U, V = ds.U.isel(time=time_index).values, ds.V.isel(time=time_index).values
        # U = U / np.sqrt(U**2 + V**2)
        # V = V / np.sqrt(U**2 + V**2)
        
        skip = 10
        ax.quiver(ds.x[::skip], ds.y[::skip], U[::skip, ::skip], V[::skip, ::skip], color="black", alpha=0.5)

    fig.tight_layout()
    return fig

def get_linechart(ds: xr.Dataset, time_index: int, ref_x_index: int, ref_y_index: int):
    """
    Plotting an interactive line chart at the location of the observation point.
    """
    days = (ds.time.values - ds.time.values[0]) / np.timedelta64(1, "D")
    time_ref = (ds.time.values[time_index] - ds.time.values[0]) / np.timedelta64(1, "D")

    U = ds.U.sel(x_index = ref_x_index, y_index = ref_y_index).values
    V = ds.V.sel(x_index = ref_x_index, y_index = ref_y_index).values
    speed = np.sqrt(U**2 + V**2)
    df = pd.DataFrame({
        "U velocity": U,
        "V velocity": V,
        "current speed": speed,
        }, index = days)
    df = (
        df.stack()
        .reset_index(level=[0, 1])
        .rename(columns={"level_0": "time (days)", "level_1": "Label", 0: "m/s"}) # Converting multiple columns into a single column
    )

    lines = alt.Chart(df).mark_line().encode(
        x = "time (days)",
        y = "m/s",
        color = "Label",
        tooltip = ["time (days)", "m/s"]
    )
    xrule = (
        alt.Chart()
        .mark_rule(color="red", strokeWidth=2)
        .encode(x=alt.datum(time_ref))
    )
    return (lines + xrule).interactive()

def render_dataset(ds: xr.Dataset):
    """
    Render data from the chosen dataset. Contains the main logic of this dashboard.
    """
    left_column, right_column = st.columns(2)
    
    with left_column.expander("🕐 Select display timestep"):
        time_index = st.slider(
            "Pick a `time` index:",
            min_value = 0,
            max_value = ds.time.shape[0] - 1,
            step = 1,
            value = int(ds.time.shape[0]/2),
            )
        timestep = dt_to_string(ds.time.values[time_index])
        st.write(f"Simulation timestep: {timestep}")

    # The point at which to plot the timeseries
    with right_column.expander("📌 Station position"):
        x = st.slider(
            "x distance:",
            min_value = ds.x[0].item(),
            max_value = ds.x[-1].item(),
            step = np.diff(ds.x.values)[0],
            value = ds.x[int(ds.x.shape[0]/2)].item(),
            )
        y = st.slider(
            "y distance:",
            min_value = ds.y[0].item(),
            max_value = ds.y[-1].item(),
            step = np.diff(ds.y.values)[0],
            value = ds.y[int(ds.y.shape[0]/2)].item(),
            )


        # Finding closest cell indices
        ref_x_index = np.argmin(np.abs(ds.x.values - x)) 
        ref_y_index = np.argmin(np.abs(ds.y.values - y))
        st.write(f"Station location at cell **x_index, y_index**: {(ref_x_index, ref_y_index)}")

    # Defining sliders for the colour ranges
    with left_column.expander("🎨 Heatmap c_ranges"):
        slider_step = 0.001
        c_ranges = {} # Instantaiting c_ranges dictionary
        color_max = float(np.abs(ds.U.values).max())
        color_slider = st.slider(
            "U endpoint",
            min_value = 0.0,
            max_value = color_max,
            value = color_max,
            step = slider_step,
            )
        c_ranges["U"] = (-color_slider, color_slider)

        color_max = float(np.abs(ds.V.values).max())
        color_slider = st.slider(
            "V endpoint",
            min_value = 0.0,
            max_value = color_max,
            value = color_max,
            step = slider_step,
            )
        c_ranges["V"] = (-color_slider, color_slider)

        color_min = float(ds.S.values.min())
        color_max = float(ds.S.values.max())
        color_slider = st.slider(
            "Sea level",
            min_value = color_min,
            max_value = color_max,
            value = (color_min, color_max),
            step = slider_step,
            )
        c_ranges["S"] = (color_slider[0], color_slider[1])

        for key in c_ranges.keys():
            st.write(f"{key}: {c_ranges[key]}")
        
    st.write("### Field snapshots")
    st.pyplot(get_figure(ds, time_index, ref_x_index, ref_y_index, c_ranges))

    st.write("### Station timeseries")
    st.altair_chart(get_linechart(ds, time_index, ref_x_index, ref_y_index), use_container_width=True)

    st.button("Export animation to MP4", on_click=export_animation, args=[ds, ref_x_index, ref_y_index, c_ranges])
    return

def export_animation(ds: xr.Dataset, ref_x_index: int, ref_y_index: int, c_ranges: dict):
    """
    Exporting the animation to MP4.
    """
    import os
    # Dealing with animation folder setup
    ani_folder = Path("temp")
    if ani_folder.exists():
        # Delete temp folder recursively if it exists
        st.warning(f"Deleting {ani_folder} folder")
        shutil.rmtree(ani_folder)
    ani_folder.mkdir()

    # Setuping up for animation
    zfill = int(np.ceil(np.log10(ds.time.values.shape[0])))
    n_timesteps = ds.time.values.shape[0]
    framerate = int(n_timesteps / 7) or 1 # ~7 seconds for animation (if framerate is 0 set to 1)

    for time_index in tqdm(range(n_timesteps), desc="Generating figures"):
        fig_name = f"{str(time_index).zfill(zfill)}.png"
        fig = get_figure(ds, time_index, ref_x_index, ref_y_index, c_ranges)
        fig.savefig(ani_folder / fig_name)
        plt.close(fig)

    # Creating animation
    pattern = f"{'/'.join(ani_folder.parts)}/%0{zfill}d.png"
    os.system(f"ffmpeg -framerate {framerate} -i {pattern} -c:v libx264 -pix_fmt yuv420p animation.mp4 -y")
    return


f = st.sidebar.file_uploader("Select a Delft3D Flow grid output file", type=(["nc"]))

if f is not None:
    st.write(f"The file is: `{f.size/1000**2:.2f}MB`")
    ds = load_data(f)
    with st.expander("Dataset information"):
        print_stats(ds)
    st.write("# Visualisations")
    render_dataset(ds)
else:
    st.error("No file selected")


