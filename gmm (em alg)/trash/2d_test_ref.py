import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

def plot_data_with_gaussian(data, confidence=0.95):
    """
    Plots 2D data points and the Gaussian distribution as an ellipse.

    Parameters:
        data (array-like): 2D data points of shape (N, 2).
        confidence (float): Confidence level for the ellipse (e.g., 0.95 for 95%).
    """
    # Compute mean and covariance of the data
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    print("Mean: ", mean)
    print("Co-variance: ", cov)

    # Compute the chi-squared value for the desired confidence interval
    chi2_val = chi2.ppf(confidence, df=2)

    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    print(eigenvalues)
    print(eigenvectors)

    order = eigenvalues.argsort()[::-1]  # Sort eigenvalues in descending order
    print(order)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    print("Eighen values: ", eigenvalues)
    print("Eighen vec: ", eigenvectors)

    # print("sss", eigenvectors[:, 0][::-1])
    # print("sss", np.arctan2(*eigenvectors[:, 0][::-1]))
    # print("sss", np.degrees(np.arctan2(*eigenvectors[:, 0][::-1])))

    # Compute ellipse parameters
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))  # Rotation angle
    width, height = 2 * np.sqrt(chi2_val * eigenvalues)        # Axis lengths

    # Plot data points
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c='blue', s=20, label='Data Points')

    # Add the ellipse
    print("AAAAAA: ", mean, mean.shape)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(ellipse)

    # Plot the mean
    ax.scatter(mean[0], mean[1], c='red', label='Mean', marker='x')

    # Adjust plot limits
    # ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
    # ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
    # ax.set_aspect('equal')
    ax.legend()
    plt.title('2D Gaussian Distribution as an Ellipse')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Example usage
# Generate random 2D data
np.random.seed(42)
data = np.random.multivariate_normal(mean=[5, 10], cov=[[3, 1], [1, 2]], size=100)

# Plot the data and its Gaussian distribution
plot_data_with_gaussian(data, confidence=0.95)
