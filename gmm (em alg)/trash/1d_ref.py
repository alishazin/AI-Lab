import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
import statistics 
import math

# np.random.seed(4)

n_samples = 300
x1 = np.random.randn(n_samples, 1) + np.array([5])
# print(x1)

x2 = np.random.randn(n_samples, 1) + np.array([15])
# print(x2)

X = np.vstack((x1, x2))
np.random.shuffle(X) 
# print(X)

plt.scatter(X[:, 0], y=np.array([0] * X.shape[0]))

K = 2

# mu = statistics.mean(x1[:, 0])
# sd = statistics.stdev(x1[:, 0])
# x_gd = np.linspace(mu - 3*sd, mu + 3*sd, 100)
# plt.plot(x_gd, norm.pdf(x_gd, mu, sd), color='red')

# mu = statistics.mean(x2[:, 0])
# sd = statistics.stdev(x2[:, 0])
# x_gd = np.linspace(mu - 3*sd, mu + 3*sd, 100)
# plt.plot(x_gd, norm.pdf(x_gd, mu, sd), color='blue')

# var = 1
var = statistics.variance(X[:, 0])
sd = math.sqrt(var/2)
print(sd)

probs = {}

point1 = X[np.random.randint(0, X.shape[0])][0]
point2 = X[np.random.randint(0, X.shape[0])][0]
print("Means: ", point1, point2)
print("Vars: ", var/2, var/2)

probs = {
    'priors' : [1/K for i in range(K)],
    'means' : [point1, point2],
    'vars' : [var/K for i in range(K)],
    'posterior' : [],
}

x_gd = np.linspace(probs['means'][0] - 3*sd, probs['means'][0] + 3*sd, 100)
plt.plot(x_gd, norm.pdf(x_gd, probs['means'][0], sd), color='red')

x_gd = np.linspace(probs['means'][1] - 3*sd, probs['means'][1] + 3*sd, 100)
plt.plot(x_gd, norm.pdf(x_gd, probs['means'][1], sd), color='blue')

plt.grid()
plt.show()



def pdf(x, mu, variance):

    if variance == 0: variance = 1e-9
    sigma = math.sqrt(variance)

    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - ((x - mu)**2) / (2 * sigma**2)
    # return (math.exp( (-(x - mu) ** 2) / (2 * (sigma**2)) )) / math.sqrt(2 * math.pi * (sigma**2))


def em_step():

    probs['posterior'] = [[], []]

    for k in range(K):

        meanNumerator = 0
        meanDenominator = 0

        varNumerator = 0
        varDenominator = 0

        for i in range(X.shape[0]):

            gammaIK = math.log(probs['priors'][k]) + pdf(X[i][0], probs['means'][k], probs['vars'][k]) - math.log(
                math.exp(
                    math.log(probs['priors'][k]) + pdf(X[i][0], probs['means'][k], probs['vars'][k])
                ) + math.exp(
                    math.log(probs['priors'][(k+1)%K]) + pdf(X[i][0], probs['means'][(k+1)%K], probs['vars'][(k+1)%K])
                )
            )
            # gammaIK = ((pdf(X[i][0], probs['means'][k], probs['vars'][k]) * probs['priors'][k]) / 
            #     (
            #         (probs['priors'][k] * pdf(X[i][0], probs['means'][k], probs['vars'][k])) +
            #         (probs['priors'][(k+1) % K] * pdf(X[i][0], probs['means'][k], probs['vars'][(k+1) % K]))
            #     )
            # )

            print(gammaIK)

            probs['posterior'][k].append(gammaIK)

            meanNumerator += math.exp(gammaIK) * X[i][0]
            meanDenominator += math.exp(gammaIK)

            varNumerator += math.exp(gammaIK) * ((X[i][0] - probs['means'][k]) ** 2)
            varDenominator += math.exp(gammaIK)

        probs['means'][k] = meanNumerator / meanDenominator
        probs['vars'][k] = varNumerator / varDenominator


def plot(k1, k2):

    print("Means: ", probs['means'][0], probs['means'][1])
    print("Vars: ", probs['vars'][0], probs['vars'][1])

    sd1 = math.sqrt(probs['vars'][0])
    sd2 = math.sqrt(probs['vars'][1])

    plt.scatter(k1, y=np.array([0] * k1.shape[0]), color='red')
    plt.scatter(k2, y=np.array([0] * k2.shape[0]), color='blue')

    x_gd = np.linspace(probs['means'][0] - 3*sd1, probs['means'][0] + 3*sd1, 100)
    plt.plot(x_gd, norm.pdf(x_gd, probs['means'][0], sd1), color='red')

    x_gd = np.linspace(probs['means'][1] - 3*sd2, probs['means'][1] + 3*sd2, 100)
    plt.plot(x_gd, norm.pdf(x_gd, probs['means'][1], sd2), color='blue')

    plt.grid()
    plt.show()


def log_likelihood():

    result = 0
    k1 = []
    k2 = []

    for i in range(X.shape[0]):

        if (probs['posterior'][0][i] > probs['posterior'][1][i]):
            k1.append(X[i][0])
        else:
            k2.append(X[i][0])

        for k in range(K):
            result += math.log(probs['priors'][k]) + pdf(X[i][0], probs['means'][k], probs['vars'][k])

    return result, np.array(k1), np.array(k2)


em_step()
res, k1, k2 = log_likelihood()
print("Log likelihood: ", res)
plot(k1, k2)

em_step()
res, k1, k2 = log_likelihood()
print("Log likelihood: ", res)
plot(k1, k2)

em_step()
res, k1, k2 = log_likelihood()
print("Log likelihood: ", res)
plot(k1, k2)

em_step()
res, k1, k2 = log_likelihood()
print("Log likelihood: ", res)
plot(k1, k2)

em_step()
res, k1, k2 = log_likelihood()
print("Log likelihood: ", res)
plot(k1, k2)