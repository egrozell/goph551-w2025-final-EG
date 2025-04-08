import devito
from examples.seismic.source import RickerSource, Receiver, TimeAxis
from src.final_project.model import demo_model
import numpy as np
import matplotlib.pyplot as plt

from sympy import init_printing, latex
init_printing(use_latex='mathjax')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

nlayers = 5
so = 8
model = demo_model(vp_top=1.0, vp_bottom=3.5, nlayers=nlayers,
                   shape=(301, 301), spacing=(10., 10.), space_order=so)
print(model.origin)
aspect_ratio = model.shape[0] / model.shape[1]
plt_config = {'cmap': 'jet', 'extent': [model.origin[0], model.origin[0] + model.domain_size[0],
                                        model.origin[1] + model.domain_size[1], model.origin[1]]}

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
slices = [slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl)]

img1 = ax[0].imshow(np.transpose(model.lam.data[slices]), vmin=1.5**2, vmax=4.0**2, **plt_config)
fig.colorbar(img1, ax=ax[0])
ax[0].set_title(r"First Lam\'e parameter $\lambda$", fontsize=20)
ax[0].set_xlabel('X (m)', fontsize=20)
ax[0].set_ylabel('Depth (m)', fontsize=20)
ax[0].set_aspect('auto')


img2 = ax[1].imshow(np.transpose(model.mu.data[slices]), vmin=0, vmax=15, **plt_config)
fig.colorbar(img2, ax=ax[1])
ax[1].set_title(r"Shear modulus $\mu$", fontsize=20)
ax[1].set_xlabel('X (m)', fontsize=20)
ax[1].set_ylabel('Depth (m)', fontsize=20)
ax[1].set_aspect('auto')


img3 = ax[2].imshow(1 / np.transpose(model.b.data[slices]), vmin=1.0, vmax=3.0, **plt_config)
fig.colorbar(img3, ax=ax[2])
ax[2].set_title(r"Density $\rho$", fontsize=20)
ax[2].set_xlabel('X (m)', fontsize=20)
ax[2].set_ylabel('Depth (m)', fontsize=20)
ax[2].set_aspect('auto')

plt.savefig("./figures/elastic_model_physical_properties.jpeg")
plt.tight_layout()
