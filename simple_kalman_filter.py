import numpy as np
from numpy.linalg import inv

x_observations = np.array([30110, 30265, 30740, 30750, 31135, 31015, 31180, 31610, 31960, 31865])
# v_observations = np.array([280, 282, 285, 286, 290])

# z = np.c_[x_observations, v_observations]
z = np.c_[x_observations]

# Initial Conditions
x_ini = 30000
v_ini = 40
t = 5  # Difference in time

# Process / Estimation Errors
error_est_x = 20
error_est_v = 5

# Observation Errors
error_obs_x = 250  # Uncertainty in the measurement
error_obs_v = 60


def prediction2d(x, v):
    A = np.array([[1, 5],
                  [0, 1]])
    X = np.array([[x],
                  [v]])
    X_prime = A.dot(X)
    return X_prime


def covariance2d(sigma1, sigma2):
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1
    cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                           [cov2_1, sigma2 ** 2]])
    return np.diag(np.diag(cov_matrix))


# Initial Estimation Covariance Matrix
P = covariance2d(error_est_x, error_est_v)
A = np.array([[1, 5],
              [0, 1]])

# Initial State Matrix
X = np.array([[x_ini],
              [v_ini]])
n = 2
H = np.identity(n)

R = covariance2d(error_obs_x, error_obs_v)

S = H.dot(P).dot(H.T) + R
K = P.dot(H).dot(inv(S))

Q = covariance2d(0.5, 0.5)

for data in z[0:]:
    newvel = float((data[0]-X[0][0])/5)
    Y = np.array([[data[0]],
                  [newvel]])

    X = prediction2d(X[0][0], X[1][0])
    P = A.dot(P).dot(A.T) + Q

    S = H.dot(P).dot(H.T) + R
    K = P.dot(H).dot(inv(S))


    X = X + K.dot(Y - X)
    P = (np.identity(len(K)) - K.dot(H)).dot(P)


    print(X)

