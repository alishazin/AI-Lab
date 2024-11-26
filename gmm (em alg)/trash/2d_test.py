
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import math
from matplotlib.patches import Ellipse
from scipy.stats import chi2

# np.random.seed(0)

n_samples = 300
x1 = np.random.randn(n_samples, 2) + np.array([5])
# print(x1)
# x3 = np.random.randn(100, 3) + np.array([3])
x2 = np.random.randn(n_samples, 2) + np.array([10])
# print(x2)

X = np.vstack((x1, x2))
# print(X)

n_components = 2

plt.scatter(X[:, 0], y=X[:, 1], marker='o', edgecolors='k', s=50)
plt.grid()
plt.show()

# X = X.transpose()

point1 = np.matrix(X[np.random.randint(0, X.shape[0])])
point2 = np.matrix(X[np.random.randint(0, X.shape[0])])

# print(point1, point2, point1.shape)

priors = [1/n_components for i in range(n_components)]
means = [point1, point2]
cov = np.cov(X, rowvar=False)
covs = [np.matrix(cov)/2 for i in range(n_components)]

# print("mean1: ", means[0].shape)
# print("cov1: ", covs[0].shape)
posterior = [[None for i in range(X.shape[0])] for k in range(n_components)]

def pdf(x, mean, cov):

    # # Computing diagonalization
    # evalues, evectors = np.linalg.eig(cov)
    # # Ensuring square root matrix exists
    # assert (evalues >= 0).all()
    # sqrt_cov = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)

    det_cov = np.linalg.det(cov) 

    # print("x - mean: ", x - mean, np.matrix(x - mean).shape)
    # print("inv: ", np.linalg.inv(cov), np.linalg.inv(cov).shape)
    # print("tr: ", (x - mean).transpose(), (x - mean).transpose().shape)

    # print(math.exp(0.5 * (np.matrix(x - mean) * np.matrix(np.linalg.inv(cov)) * np.matrix((x - mean).transpose())).item(0,0)) )
    # print((2*math.pi) * det_cov)

    return (math.exp(-0.5 * (np.matrix(x - mean) * np.matrix(np.linalg.inv(cov)) * np.matrix(np.matrix(x - mean).transpose())) )) / ((2*math.pi) * math.sqrt(det_cov))


def log_pdf(x, mean, cov):

    det_cov = np.linalg.det(cov) 

    return -1 * math.log(2 * math.pi) - 0.5 * math.log(det_cov) - 0.5 * ( np.matrix(x - mean) * np.linalg.inv(cov) * np.matrix(x - mean).transpose() )


def e_step():

    global posterior

    for k in range(n_components):

        for i in range(X.shape[0]):

            denominator_value = 0
            for j in range(n_components):
                denominator_value += priors[j] * pdf(X[i], means[j], covs[j])
            denominator_value = denominator_value

            posterior[k][i] = (priors[k] * pdf(X[i], means[k], covs[k])) / denominator_value

    # print(posterior)
    # print(posterior[-1], posterior[-1])


def m_step():

    for k in range(n_components):

        mean_numerator = 0
        cov_numerator = 0
        denominator = 0

        for i in range(X.shape[0]):

            denominator += posterior[k][i]
            mean_numerator += np.matrix(X[i]) * posterior[k][i]
            # print("cov_numerator: ", np.matrix(X[i]).shape)
            # print("cov_numerator: ", means[k].shape)
            cov_numerator += posterior[k][i] * (np.matrix(np.matrix(X[i]) - means[k]).transpose() * np.matrix(np.matrix(X[i]) - means[k]))

        means[k] = np.matrix(mean_numerator) / denominator
        covs[k] = np.matrix(cov_numerator) / denominator

        print("means: ", means)
        print("covs: ", covs)


def plot_data_with_gaussian(k1, k2, confidence=0.95):

    # Plot data points
    fig, ax = plt.subplots()
    ax.scatter(k1[:, 0], k1[:, 1], c='red', s=20, label='Data Points')
    ax.scatter(k2[:, 0], k2[:, 1], c='blue', s=20, label='Data Points')

    colors = ['red', 'blue']

    for k in range(n_components):

        # Compute mean and covariance of the data
        mean = means[k]
        # print("Mean: ", mean, mean.shape)
        cov = covs[k]

        # Compute the chi-squared value for the desired confidence interval
        chi2_val = chi2.ppf(confidence, df=2)

        # Eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order = eigenvalues.argsort()[::-1]  # Sort eigenvalues in descending order
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Compute ellipse parameters
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))  # Rotation angle
        width, height = 2 * np.sqrt(chi2_val * eigenvalues)        # Axis lengths

        # Add the ellipse
        # print(mean.item(0,0))
        # print(100)
        ellipse = Ellipse(xy=np.array([mean.item(0, 0), mean.item(0, 1)]), width=width, height=height, angle=angle, edgecolor=colors[k], facecolor='none', lw=2)
        ax.add_patch(ellipse)

        # Plot the mean
        # print(mean[0], mean[1])
        ax.scatter(mean.item(0, 0), mean.item(0, 1), c=colors[k], label='Mean', marker='x')

    # Adjust plot limits
    # ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
    # ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
    # ax.set_aspect('equal')
    ax.legend()
    plt.title('2D Gaussian Distribution as an Ellipse')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


def log_likelihood():

    result = 0
    k1 = []
    k2 = []

    for i in range(X.shape[0]):

        if (posterior[0][i] > posterior[1][i]):
            k1.append(X[i])
        else:
            k2.append(X[i])

        for k in range(n_components):
            result += math.log(priors[k]) + log_pdf(X[i], means[k], covs[k]).item(0, 0)

    return np.array(k1), np.array(k2), result


for i in range(10):
    e_step()
    m_step()
    k1, k2, res = log_likelihood()
    print("Result: ", res)
    plot_data_with_gaussian(k1, k2)