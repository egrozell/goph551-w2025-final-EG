import numpy as np
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
import gempy as gp
import gempy_viewer as gpv
import devito as dv
from examples.seismic import TimeAxis
from examples.seismic import Model
from examples.seismic import RickerSource
from examples.seismic import Receiver
# import pyvista as pv
display = Display(visible=0, size=(600, 400))
display.start()


def create_surface(model, points, surface):
    """Add a list of points to a surface in a model"""
    kwargs = {'geo_model': model,
              'x': [x[0] for x in points],
              'y': [y[1] for y in points],
              'z': [z[2] for z in points],
              'elements_names': [surface for i in range(len(points))]}
    print(kwargs)
    gp.add_surface_points(**kwargs)


def main():
    # Setting model params
    extent = (-5., 1005., -5., 1005., -1005., 5.)
    shape = (101, 101, 101)
    geo_model = gp.create_geomodel(
        project_name='Gempy', extent=extent, resolution=shape,
        importer_helper=gp.data.ImporterHelper(
            path_to_surface_points="./model_points.csv",
            path_to_orientations="./orientations.csv"
        )
    )

    # geo_model.surfaces
    sol = gp.compute_model(geo_model)
    # gpv.plot_2d(geo_model, show_lith=False, show_boundaries=False)

    print("sol.block_solution_type")
    print(sol.block_solution_type)
    print("sol.debug_input_data")
    print(sol.debug_input_data)
    print("sol.gravity")
    print(sol.gravity)
    print("sol.magnetics")
    print(sol.magnetics)
    print("sol.raw_arrays")
    print(sol.raw_arrays)
    print("sol.scalar_field_at_surface_points")
    print(sol.scalar_field_at_surface_points)
    print("sol.octrees_output")
    print(sol.octrees_output)
    print("sol.dc_meshes")
    print(sol.dc_meshes)
    print("sol.block_solution_type")
    print(sol.block_solution_type)

    # Reshaping our data to the shape required by Devito
    reshaped = np.reshape(sol.values_matrix, shape, order='C')
    reshaped.shape

    seis_model = Model(vp=reshaped,
                       origin=(0., 0., -1000.),
                       spacing=(10., 10., 10.),
                       shape=shape, nbl=30,
                       space_order=4,
                       bcs="damp")

    t0 = 0.  # Simulation starts a t=0
    tn = 1000.  # Simulation last 1 second (1000 ms)
    dt = seis_model.critical_dt  # Time step from model grid spacing

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.015  # Source peak frequency is 15Hz (0.015 kHz)
    src = RickerSource(name='src', grid=seis_model.grid, f0=f0,
                       npoint=1, time_range=time_range)

    # First, position source centrally in all dimensions, then set depth
    src.coordinates.data[:] = np.array(seis_model.domain_size) * .5
    src.coordinates.data[0, -1] = -20  # Depth is 20m

    # We can plot the time signature to see the wavelet
    src.show()

    # Create symbol for 101 receivers
    rec = Receiver(name='rec', grid=seis_model.grid, npoint=101, time_range=time_range)

    # Prescribe even spacing for receivers along the x-axis
    rec.coordinates.data[:, 0] = np.linspace(0, seis_model.domain_size[0], num=101)
    rec.coordinates.data[:, 1] = 0.5 * seis_model.domain_size[1]
    rec.coordinates.data[:, -1] = -20.  # Depth is 20m
    # Define the wavefield with the size of the model and the time dimension
    u = dv.TimeFunction(name="u", grid=seis_model.grid, time_order=2, space_order=4)

    # We can now write the PDE
    pde = seis_model.m * u.dt2 - u.laplace + seis_model.damp * u.dt

    # The PDE representation is as on paper
    pde

    # This discrete PDE can be solved in a time-marching way updating u(t+dt) from the previous time
    # step Devito as a shortcut for u(t+dt) which is u.forward. We can then rewrite the
    # PDE as a time marching updating equation known as a stencil using customized SymPy functions

    stencil = dv.Eq(u.forward, dv.solve(pde, u.forward))
    stencil

    # Finally we define the source injection and
    # receiver read function to generate the corresponding code
    src_term = src.inject(field=u.forward, expr=src * dt**2 / seis_model.m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u.forward)

    op = dv.Operator([stencil] + src_term + rec_term, subs=seis_model.spacing_map)

    op(time=time_range.num - 1, dt=seis_model.critical_dt)

    plt.imshow(rec.data, cmap='viridis', aspect='auto', vmax=0.01, vmin=-0.01)
    plt.xlabel("Reciever number")
    plt.ylabel("Time (ms)")
    plt.colorbar()
    plt.show()

    # Set default pyvista backend
    pv.set_jupyter_backend('ipyvtklink')

    # Trim down the data from u to remove damping field
    trimmed_data = u.data[1, 30:-30, 30:-30, 30:-30]

    # Create the spatial reference
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(trimmed_data.shape) + 1

    # Edit the spatial reference
    grid.origin = (0., 0., -1000.)  # The bottom left corner of the data set
    grid.spacing = (10, 10, 10)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["values"] = trimmed_data.flatten(order="F")  # Flatten the array!

    orth_slices = grid.slice_orthogonal(x=200, y=200, z=-500)

    orth_slices.plot(cmap='seismic', clim=[-0.01, 0.01])

    y_slices = grid.slice_along_axis(n=5, axis="y")
    p = pv.Plotter()
    p.add_mesh(grid.outline(), color="k")
    p.add_mesh(y_slices, cmap='seismic', clim=[-0.01, 0.01], opacity=0.8)
    p.show()


if __name__ == "__main__":
    main()
