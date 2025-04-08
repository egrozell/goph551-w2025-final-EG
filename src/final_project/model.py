from examples.seismic.model import SeismicModel
import numpy as np


def Gardners(vp, normalize=True):
    """
    Gardner's relation for vp in km/s
    """
    b = 1 / (0.31 * (1e3 * vp)**0.25)
    if normalize:
        b[vp < 1.51] = 1.0
    return b


def demo_model(vp_top=1.5, vp_bottom=3.5, **kwargs):
    """
    inputs
    ______
    vp_top: the p velocity for the top of the model
    vp_bottom : the p velocity for the bottom of the model
    **kwargs: following additional arguements
    space_order: the dimension of the model this case 2d
    shape: number of grid points (nx, nz)
    spacing: grid spacing in m
    origin: the location of the top left corner
    nbl: Number of outer layers (such as absorbing layers for boundary damping)
    nlayers: Number of layers for the model

    outputs
    _______
    space_order: the dimension of the model
    vp: the p velocity values
    vs: the s velocity values
    b: the bulk density values
    spacing: grid spacing in m
    origin: the location of the top left corner
    nbl: Number of outer layers (such as absorbing layers for boundary damping)
    **kwargs: any additional arguements

    raises
    ______
    """
    space_order = kwargs.pop('space_order', 2)
    shape = kwargs.pop('shape', (101, 101))
    spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
    origin = kwargs.pop('origin', tuple([0. for _ in shape]))
    nbl = kwargs.pop('nbl', 10)
    dtype = kwargs.pop('dtype', np.float32)
    nlayers = kwargs.pop('nlayers', 3)

    # A n-layers model in a 2D or 3D domain with two different
    # velocities split across the height dimension:
    # By default, the top part of the domain has 1.5 km/s,
    # and the bottom part of the domain has 2.5 km/s.
    vp_top = kwargs.pop('vp_top', vp_top)
    vp_bottom = kwargs.pop('vp_bottom', vp_bottom)
    # Define a velocity profile in km/s
    v = np.empty(shape, dtype=dtype)
    v[:] = vp_top  # Top velocity (background)
    vp_i = np.linspace(vp_top, vp_bottom, nlayers)
    for i in range(1, nlayers):
        v[..., i * int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

        vs = 0.5 * v[:]
        b = Gardners(v)
        vs[v < 1.51] = 0.0

    return SeismicModel(space_order=space_order, vp=v, vs=vs, b=b,
                        origin=origin, shape=shape,
                        dtype=dtype, spacing=spacing, nbl=nbl, **kwargs)


