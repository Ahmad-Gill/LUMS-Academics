import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [2, 0],
    [0, 2],
    [1, -1],
    [-1, -1],
    [-2, 0]
])

pc1 = np.array([1, 0])
X_proj = X @ pc1
proj = np.outer(X_proj, pc1)

plt.scatter(X[:, 0], X[:, 1], color='blue', label='Original Data')
plt.quiver(0, 0, pc1[0], pc1[1], color='red', scale=1, scale_units='xy', angles='xy', label='1st PC')
plt.scatter(proj[:, 0], proj[:, 1], color='orange', label='Projected Points')
for i in range(len(X)):
    plt.plot([X[i, 0], proj[i, 0]], [X[i, 1], proj[i, 1]], 'k--', alpha=0.5)
plt.axis('equal')
plt.title('Projection on First Principal Component (62.5% Variance Retained)')
plt.legend()
plt.savefig("pca_projection.png")
