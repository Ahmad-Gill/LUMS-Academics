import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Random vectors in R^3
v1 = np.random.randn(3)
v2 = np.random.randn(3)
v3 = np.random.randn(3)

# Parameter ranges
t = np.linspace(-1, 1, 50)

# Combinations
linear = np.array([a*v1 + b*v2 + c*v3 for a in t for b in t for c in t])
affine = np.array([(1-a-b)*v1 + a*v2 + b*v3 for a in t for b in t])
convex = np.array([(1-a-b)*v1 + a*v2 + b*v3 for a in np.linspace(0,1,30)
                   for b in np.linspace(0,1,30) if a+b <= 1])

# -------- FIGURE 1: LINEAR --------
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title("Linear Combinations")
ax1.scatter(linear[:,0], linear[:,1], linear[:,2], s=2)
plt.show()

# -------- FIGURE 2: AFFINE --------
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Affine Combinations")
ax2.scatter(affine[:,0], affine[:,1], affine[:,2], s=2)
plt.show()

# -------- FIGURE 3: CONVEX --------
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_title("Convex Combinations")
ax3.scatter(convex[:,0], convex[:,1], convex[:,2], s=2)
plt.show()

# -------- FIGURE 4: ALL COMBINED --------
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title("Combined: Linear, Affine, Convex")
ax4.scatter(linear[:,0], linear[:,1], linear[:,2], s=2)
ax4.scatter(affine[:,0], affine[:,1], affine[:,2], s=2)
ax4.scatter(convex[:,0], convex[:,1], convex[:,2], s=2)
plt.show()
