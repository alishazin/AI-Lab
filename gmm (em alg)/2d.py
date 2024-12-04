
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Ellipse
from scipy.stats import chi2


class GaussianMixtureModel:

    def __init__(self, n_components, max_iter=100, threshold=10e-4, log_initial_step=False, log_steps=False, fixed_priors=True, confidence=0.95):

        self.n_components = n_components
        self.max_iter = max_iter
        self.threshold = threshold
        self.log_initial_step = log_initial_step
        self.log_steps = log_steps
        self.fixed_priors = fixed_priors
        self.confidence = confidence

        self.priors = []
        self.means = []
        self.covs = []
        self.prev_priors = []
        self.prev_means = []
        self.prev_covs = []
        self.prev_log_likelihood = None
        self.posteriors = None
        self.iter_count = None


    def fit(self, X):

        cov = np.cov(X, rowvar=False)

        for k in range(self.n_components):

            self.covs.append(np.matrix(cov)/self.n_components)
            self.means.append(np.matrix(X[np.random.randint(0, X.shape[0])]))
            self.priors.append(1/self.n_components)

        if self.log_initial_step:

            color = iter(cm.viridis(np.linspace(0, 1, self.n_components)))

            fig, ax = plt.subplots()
            plt.scatter(X[:, 0], y=X[:, 1], marker='o', edgecolors='k', s=50)

            for k in range(self.n_components):
                
                c = next(color)

                mean = self.means[k]
                cov = self.covs[k]

                ax.scatter(mean.item(0, 0), mean.item(0, 1), color=c, label=f'k={k+1}', marker='o')

                chi2_val = chi2.ppf(self.confidence, df=2)

                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

                angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                width, height = 2 * np.sqrt(chi2_val * eigenvalues)

                ellipse = Ellipse(xy=np.array([mean.item(0, 0), mean.item(0, 1)]), width=width, height=height, angle=angle, edgecolor=c, facecolor='none', lw=2)
                ax.add_patch(ellipse)

            plt.grid()
            plt.title("Initial State")
            plt.legend()
            plt.show()

        
    def predict(self, X):

        self.posteriors = [[None for i in range(X.shape[0])] for k in range(self.n_components)]
        labels = None

        for i in range(self.max_iter):

            self.iter_count = i+1

            self.e_step(X)
            self.m_step(X)

            if self.check_global_convergence(X):
                break

            labels = self.get_labels(X)

            if self.log_steps:
                self.plot(X, labels, final=False)
            
        return labels


    def e_step(self, X):

        for k in range(self.n_components):

            for i in range(X.shape[0]):

                denominator_value = 0
                for j in range(n_components):
                    denominator_value += self.priors[j] * self.pdf(X[i], self.means[j], self.covs[j])
                denominator_value = denominator_value

                self.posteriors[k][i] = (self.priors[k] * self.pdf(X[i], self.means[k], self.covs[k])) / denominator_value


    def m_step(self, X):

        self.prev_priors = self.priors.copy()
        self.prev_means = self.means.copy()
        self.prev_covs = self.covs.copy()

        for k in range(self.n_components):

            mean_numerator = 0
            cov_numerator = 0
            denominator = 0

            for i in range(X.shape[0]):

                denominator += self.posteriors[k][i]
                mean_numerator += np.matrix(X[i]) * self.posteriors[k][i]
                cov_numerator += self.posteriors[k][i] * (np.matrix(np.matrix(X[i]) - self.means[k]).transpose() * np.matrix(np.matrix(X[i]) - self.means[k]))

            if not self.fixed_priors:
                self.priors[k] = denominator / X.shape[0]

            self.means[k] = np.matrix(mean_numerator) / denominator
            self.covs[k] = np.matrix(cov_numerator) / denominator


    def check_global_convergence(self, X):

        prior_diff = np.abs(np.array(self.prev_priors) - np.array(self.priors)).max()
        mean_diff = np.linalg.norm(np.array(self.prev_means) - np.array(self.means), axis=1).max()
        cov_diff = max([np.linalg.norm(cov - cov_prev, ord='fro') 
                    for cov, cov_prev in zip(self.covs, self.prev_covs)]) # (Frobenius norm)

        temp_log_likelikood = self.log_likelihood(X)
        log_likelikood_diff = abs(self.log_likelihood(X) - self.prev_log_likelihood) if self.prev_log_likelihood != None else None
        self.prev_log_likelihood = temp_log_likelikood
            
        return (prior_diff < self.threshold) and (mean_diff < self.threshold) and (cov_diff < self.threshold) and (log_likelikood_diff != None) and (log_likelikood_diff < self.threshold)
    

    def log_likelihood(self, X):

        result = 0

        for i in range(X.shape[0]):
            for k in range(self.n_components):
                result += math.log(self.priors[k]) + self.log_pdf(X[i], self.means[k], self.covs[k]).item(0, 0)

        return result
    

    def get_labels(self, X):

        labels = []

        for i in range(X.shape[0]):

            values = [self.posteriors[k][i] for k in range(self.n_components)]
            labels.append(max(range(len(values)), key=values.__getitem__))

        return labels


    def plot(self, X, labels, confidence=0.95, final=True):

        separated_X = [[] for k in range(self.n_components)]

        for index, label in enumerate(labels):
            separated_X[label].append(X[index])

        color = iter(cm.viridis(np.linspace(0, 1, self.n_components)))
        fig, ax = plt.subplots()

        for k in range(self.n_components):

            c = next(color) 

            # Compute mean and covariance of the data
            mean = self.means[k]
            cov = self.covs[k]

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

            ellipse = Ellipse(xy=np.array([mean.item(0, 0), mean.item(0, 1)]), width=width, height=height, angle=angle, edgecolor=c, facecolor='none', lw=2)
            ax.add_patch(ellipse)

            ax.scatter(mean.item(0, 0), mean.item(0, 1), color=c, marker='x')
            ax.scatter([i[0] for i in separated_X[k]], [i[1] for i in separated_X[k]], color=c, s=20, label=f"k={k+1}")

        if final:
            plt.title("Final State")
        else:
            plt.title(f"After iteration {self.iter_count}")

        plt.legend()
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    
    @classmethod
    def pdf(cls, x, mean, cov):

        det_cov = np.linalg.det(cov) 

        return (math.exp(-0.5 * (np.matrix(x - mean) * np.matrix(np.linalg.inv(cov)) * np.matrix(np.matrix(x - mean).transpose())) )) / ((2*math.pi) * math.sqrt(det_cov))
    

    @classmethod
    def log_pdf(cls, x, mean, cov):

        det_cov = np.linalg.det(cov) 

        return -1 * math.log(2 * math.pi) - 0.5 * math.log(det_cov) - 0.5 * ( np.matrix(x - mean) * np.linalg.inv(cov) * np.matrix(x - mean).transpose() )



np.random.seed(100)

n_samples = 300
x1 = np.random.randn(n_samples, 2) + np.array([5])
x2 = np.random.randn(n_samples, 2) + np.array([10])

X = np.vstack((x1, x2))
np.random.shuffle(X)

n_components = 2

gmm = GaussianMixtureModel(n_components=n_components, log_steps=True, log_initial_step=True, fixed_priors=True)
gmm.fit(X)

labels = gmm.predict(X)
gmm.plot(X, labels)