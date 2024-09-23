import torch
import torch.linalg as linalg
import matplotlib.pyplot as plt


mean_A = torch.tensor([-0.5, -0.5])
cov_A = torch.tensor([[1, 0.25], [0.25, 1]])
dist_A = torch.distributions.MultivariateNormal(mean_A, cov_A)
samples_A = dist_A.sample((1000,))

mean_B = torch.tensor([0.5, 0.5])
cov_B = torch.tensor([[1, 0.25], [0.25, 1]])
dist_B = torch.distributions.MultivariateNormal(mean_B, cov_B)
samples_B = dist_B.sample((1000,))
plt.scatter(samples_A[:, 0], samples_A[:, 1], label='A', alpha=0.6)
plt.scatter(samples_B[:, 0], samples_B[:, 1], label='B', alpha=0.6)
plt.legend()
plt.show()
X = torch.cat([samples_A, samples_B], dim=0)

def kmeans(X, k=2, max_iters=100):
    centroids = X[torch.randperm(X.size(0))[:k]]
    for _ in range(max_iters):
        distances = torch.cdist(X, centroids)
        clusters = distances.argmin(dim=1)
        new_centroids = torch.stack([X[clusters == i].mean(0) for i in range(k)])
        if torch.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids
clusters, centroids = kmeans(X)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red')
plt.show()
from torch.linalg import svd
X_centered = X - X.mean(0)
U, S, Vt = svd(X_centered)
X_pca = X_centered @ Vt.T[:, :2]
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title('PCA projection')
plt.show()
