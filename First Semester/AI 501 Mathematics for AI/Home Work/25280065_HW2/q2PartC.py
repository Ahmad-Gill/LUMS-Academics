import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 200)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

diamond_x = [1, 0, -1, 0, 1]
diamond_y = [0, 1, 0, -1, 0]

plt.plot(circle_x, circle_y, 'r', label='L2 (Ridge)')
plt.plot(diamond_x, diamond_y, 'b', label='L1 (Lasso)')

plt.title('Geometric Intuition: L1 vs L2 Constraints')
plt.xlabel('w1')
plt.ylabel('w2')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
