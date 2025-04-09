import devito
from examples.seismic.source import RickerSource, Receiver, TimeAxis
# from src.final_project.model import demo_model
import final_project.model_from_bin as bin_model
import numpy as np
import matplotlib.pyplot as plt

from sympy import init_printing, latex
init_printing(use_latex='mathjax')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

dir = "./models/Marmousi/"
so = 8
model = bin_model.model(dir, spacing=(4., 4.), space_order=so)
# vp_min, vp_max, vs_min, vs_max, rho_min, rho_max = bin_model.get_max_min(dir)
print(model.origin)
aspect_ratio = model.shape[0] / model.shape[1]
plt_config = {'cmap': 'RdBu', 'extent': [model.origin[0], model.origin[0] + model.domain_size[0],
                                        model.origin[1] + model.domain_size[1], model.origin[1]]}

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
slices = [slice(model.nbl, -model.nbl), slice(model.nbl, -model.nbl)]
scale = 2
img1 = ax[0].imshow(np.transpose(model.lam.data[slices]), vmin=1500, vmax=4500, **plt_config)
fig.colorbar(img1, ax=ax[0])
ax[0].set_title(r"First Lam\'e parameter $\lambda$", fontsize=20)
ax[0].set_xlabel('X (m)', fontsize=20)
ax[0].set_ylabel('Depth (m)', fontsize=20)
ax[0].set_aspect('auto')


img2 = ax[1].imshow(np.transpose(model.mu.data[slices]), vmin=400, vmax=3000, **plt_config)
fig.colorbar(img2, ax=ax[1])
ax[1].set_title(r"Shear modulus $\mu$", fontsize=20)
ax[1].set_xlabel('X (m)', fontsize=20)
ax[1].set_ylabel('Depth (m)', fontsize=20)
ax[1].set_aspect('auto')


img3 = ax[2].imshow(1 / np.transpose(model.b.data[slices]), vmin=1.9, vmax=2.8, **plt_config)
fig.colorbar(img3, ax=ax[2])
ax[2].set_title(r"Density $\rho$", fontsize=20)
ax[2].set_xlabel('X (m)', fontsize=20)
ax[2].set_ylabel('Depth (m)', fontsize=20)
ax[2].set_aspect('auto')

plt.savefig("./figures/Marmousi_elastic_model_physical_properties.tiff")
plt.tight_layout()
