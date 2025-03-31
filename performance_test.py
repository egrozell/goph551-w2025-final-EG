# Some imports we will need below
import numpy as np
from devito import *
import matplotlib.pyplot as plt 
nx, ny = 100, 100
grid = Grid(shape=(nx, ny))
u = TimeFunction(name='u', grid=grid, space_order=2, save=200)
c = Constant(name='c')
eqn = Eq(u.dt, c * u.laplace)
step = Eq(u.forward, solve(eqn, u.forward))
op = Operator([step])
xx, yy = np.meshgrid(np.linspace(0., 1., nx, dtype=np.float32),
                     np.linspace(0., 1., ny, dtype=np.float32))
r = (xx - .5)**2. + (yy - .5)**2.
# Inserting the ring
u.data[0, np.logical_and(.05 <= r, r <= .1)] = 1.
stats = op.apply(dt=5e-05, c=0.5)
plt.rcParams['figure.figsize'] = (20, 20)
for i in range(1, 6):
    plt.subplot(1, 6, i)
    plt.imshow(u.data[(i-1)*40])
plt.savefig("figures/figure.svg")
