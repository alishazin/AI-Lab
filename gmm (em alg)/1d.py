
import statistics
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import norm 


class GaussianMixtureModel:

    def __init__(self, n_components, max_iter=100, threshold=10e-4, log_initial_step=False, log_steps=False, fixed_priors=True):

        self.n_components = n_components
        self.max_iter = max_iter
        self.threshold = threshold
        self.log_initial_step = log_initial_step
        self.log_steps = log_steps
        self.fixed_priors = fixed_priors

        self.priors = []
        self.means = []
        self.vars = []
        self.prev_priors = []
        self.prev_means = []
        self.prev_vars = []
        self.prev_log_likelihood = None
        self.posteriors = None
        self.iter_count = None


    def fit(self, X):

        var = statistics.variance(X[:, 0])

        for k in range(self.n_components):

            self.vars.append(var/self.n_components)
            self.means.append(X[np.random.randint(0, X.shape[0])][0])
            self.priors.append(1/self.n_components)

        if self.log_initial_step:

            color = iter(cm.viridis(np.linspace(0, 1, self.n_components)))

            plt.scatter(X, y=np.array([0] * X.shape[0]))

            for k in range(self.n_components):

                c = next(color) 
                sd = math.sqrt(self.vars[k])
                x_gd = np.linspace(self.means[k] - 3*sd, self.means[k] + 3*sd, 100)
                plt.plot(x_gd, norm.pdf(x_gd, self.means[k], sd), color=c, label=f"k={k+1}")

            plt.legend()
            plt.grid()
            plt.title("Initial State")
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
                for j in range(self.n_components):
                    denominator_value += math.exp(
                        math.log(self.priors[j]) + self.log_pdf(X[i][0], self.means[j], self.vars[j])
                    )
                denominator_value = math.log(denominator_value)

                self.posteriors[k][i] = math.log(self.priors[k]) + self.log_pdf(X[i][0], self.means[k], self.vars[k]) - denominator_value


    def m_step(self, X):

        self.prev_priors = self.priors.copy()
        self.prev_means = self.means.copy()
        self.prev_vars = self.vars.copy()

        for k in range(self.n_components):    

            prior_numerator = 0 # same for mean denominator and variance denominator
            mean_numerator = 0
            var_numerator = 0

            for i in range(X.shape[0]):

                exp_value = math.exp(self.posteriors[k][i])

                prior_numerator += exp_value
                mean_numerator += exp_value * X[i][0]
                var_numerator += exp_value * ((X[i][0] - self.means[k]) ** 2)

            if not self.fixed_priors:
                self.priors[k] = prior_numerator / X.shape[0]

            self.means[k] = mean_numerator / prior_numerator
            self.vars[k] = var_numerator / prior_numerator
    
    
    def check_global_convergence(self, X):

        prior_diff = np.abs(np.array(self.prev_priors) - np.array(self.priors)).max()
        mean_diff = np.abs(np.array(self.prev_means) - np.array(self.means)).max()
        var_diff = np.abs(np.array(self.prev_vars) - np.array(self.vars)).max()

        temp_log_likelikood = self.log_likelihood(X)
        log_likelikood_diff = abs(self.log_likelihood(X) - self.prev_log_likelihood) if self.prev_log_likelihood != None else None
        self.prev_log_likelihood = temp_log_likelikood
            
        return (prior_diff < self.threshold) and (mean_diff < self.threshold) and (var_diff < self.threshold) and (log_likelikood_diff != None) and (log_likelikood_diff < self.threshold)
    

    def log_likelihood(self, X):

        result = 0

        for i in range(X.shape[0]):

            for k in range(self.n_components):
                result += math.log(self.priors[k]) + self.log_pdf(X[i][0], self.means[k], self.vars[k])

        return result
    

    def get_labels(self, X):

        labels = []

        for i in range(X.shape[0]):

            values = [self.posteriors[k][i] for k in range(self.n_components)]
            labels.append(max(range(len(values)), key=values.__getitem__))

        return labels


    def plot(self, X, labels, final=True):

        separated_X = [[] for k in range(self.n_components)]

        for index, label in enumerate(labels):
            separated_X[label].append(X[index][0])

        color = iter(cm.viridis(np.linspace(0, 1, self.n_components)))

        for k in range(self.n_components):

            c = next(color) 

            sd = math.sqrt(self.vars[k])

            plt.scatter(separated_X[k], y=np.array([0] * len(separated_X[k])), color=c, label=f"k={k+1}")

            x_gd = np.linspace(self.means[k] - 3*sd, self.means[k] + 3*sd, 100)
            plt.plot(x_gd, norm.pdf(x_gd, self.means[k], sd), color=c)

        if final:
            plt.title("Final State")
        else:
            plt.title(f"After iteration {self.iter_count}")

        plt.legend()
        plt.grid()
        plt.show()

    
    @classmethod
    def pdf(cls, x, mu, variance):

        if variance == 0: 
            variance = 1e-9

        sigma = math.sqrt(variance)

        return (math.exp( (-(x - mu) ** 2) / (2 * (sigma**2)) )) / math.sqrt(2 * math.pi * (sigma**2))
    

    @classmethod
    def log_pdf(cls, x, mu, variance):

        if variance == 0: 
            variance = 1e-9

        sigma = math.sqrt(variance)

        return -0.5 * np.log(2 * np.pi) - np.log(sigma) - ((x - mu)**2) / (2 * sigma**2)

# np.random.seed(4)

n_samples = 300
x1 = np.random.randn(n_samples, 1) + np.array([5])
x2 = np.random.randn(n_samples, 1) + np.array([10])
x3 = np.random.randn(n_samples, 1) + np.array([15])

X = np.vstack((x1, x2, x3))
np.random.shuffle(X)

n_components = 3

gmm = GaussianMixtureModel(n_components=n_components, log_steps=False, log_initial_step=True, fixed_priors=True)
gmm.fit(X)
labels = gmm.predict(X)

gmm.plot(X, labels)
print(gmm.iter_count)