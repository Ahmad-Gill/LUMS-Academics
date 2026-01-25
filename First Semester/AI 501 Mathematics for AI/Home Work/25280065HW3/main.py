import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# Step 0: Prepare dataset
# ----------------------------
# Negative points (-1)
neg_points = np.array([[-1,0], [1,1], [1.5,1], [2,1], [1.5,2]])
neg_labels = -np.ones(len(neg_points))

# Positive points (+1)
pos_points = np.array([[2,0], [0,1], [3,1], [0,2], [2,3]])
pos_labels = np.ones(len(pos_points))

# Combine
X = np.vstack([neg_points, pos_points])
y = np.hstack([neg_labels, pos_labels])
N = X.shape[0]

# ----------------------------
# Step 1: Hinge-loss Gradient Descent
# ----------------------------
def hinge_loss_gradient_descent(X, y, lr=0.01, epochs=5000):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        grad_w = np.zeros_like(w)
        grad_b = 0.0
        for i in range(N):
            margin = y[i] * (np.dot(w, X[i]) + b)
            if margin < 1:
                grad_w -= y[i] * X[i]
                grad_b -= y[i]
        w -= lr * grad_w / N
        b -= lr * grad_b / N
    return w, b

# ----------------------------
# Step 2: Fit linear classifier in 2D
# ----------------------------
w_2d, b_2d = hinge_loss_gradient_descent(X, y)
margins_2d = y * (X @ w_2d + b_2d)
misclassified_2d = np.sum(margins_2d < 0)
accuracy_2d = np.mean(margins_2d >= 0) * 100

print("2D Classifier Results:")
print("Weights:", w_2d)
print("Bias:", b_2d)
print("Misclassified points:", misclassified_2d)
print("Training Accuracy: {:.2f}%".format(accuracy_2d))

# ----------------------------
# Step 3: Transform to 3D using feature map
# ----------------------------
def feature_map(X):
    x1 = X[:,0]
    x2 = X[:,1]
    X_new = np.column_stack([x1**2, x2**2, np.sqrt(2)*x1*x2])
    return X_new

X_3d = feature_map(X)

# ----------------------------
# Step 4: Fit linear classifier in 3D
# ----------------------------
w_3d, b_3d = hinge_loss_gradient_descent(X_3d, y)
margins_3d = y * (X_3d @ w_3d + b_3d)
misclassified_3d = np.sum(margins_3d < 0)
accuracy_3d = np.mean(margins_3d >= 0) * 100

print("\n3D Classifier Results:")
print("Weights:", w_3d)
print("Bias:", b_3d)
print("Misclassified points:", misclassified_3d)
print("Training Accuracy: {:.2f}%".format(accuracy_3d))

# ----------------------------
# Step 5: Plot 2D classifier
# ----------------------------
plt.figure(figsize=(7,5))
plt.scatter(pos_points[:,0], pos_points[:,1], c='blue', label='Positive (+1)', marker='^')
plt.scatter(neg_points[:,0], neg_points[:,1], c='red', label='Negative (-1)', marker='o')

# Decision boundary line in 2D: w1*x + w2*y + b = 0 => y = (-w1*x - b)/w2
if w_2d[1] != 0:
    x_vals = np.linspace(-1, 4, 100)
    y_vals = (-w_2d[0]*x_vals - b_2d)/w_2d[1]
    plt.plot(x_vals, y_vals, 'k--', label='2D Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Linear Classifier')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Step 6: Plot 3D classifier
# ----------------------------
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

# Plot positive points
pos_3d = X_3d[y==1]
ax.scatter(pos_3d[:,0], pos_3d[:,1], pos_3d[:,2], c='blue', marker='^', label='Positive (+1)')

# Plot negative points
neg_3d = X_3d[y==-1]
ax.scatter(neg_3d[:,0], neg_3d[:,1], neg_3d[:,2], c='red', marker='o', label='Negative (-1)')

# Create grid for plane
xx, yy = np.meshgrid(np.linspace(np.min(X_3d[:,0])-0.5, np.max(X_3d[:,0])+0.5, 10),
                     np.linspace(np.min(X_3d[:,1])-0.5, np.max(X_3d[:,1])+0.5, 10))
# Plane equation: w1*x + w2*y + w3*z + b = 0 => z = (-w1*x - w2*y - b)/w3
if w_3d[2] != 0:
    zz = (-w_3d[0]*xx - w_3d[1]*yy - b_3d)/w_3d[2]
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

ax.set_xlabel('x1^2')
ax.set_ylabel('x2^2')
ax.set_zlabel('sqrt(2)*x1*x2')
ax.set_title('3D Linear Classifier in Transformed Space')
ax.legend()
plt.show()
