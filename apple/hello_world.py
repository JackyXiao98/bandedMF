
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from pfl.internal.ops import pytorch_ops
from pfl.internal.ops.selector import set_framework_module

os.environ['PFL_PYTORCH_DEVICE'] = 'cpu'
set_framework_module(pytorch_ops)

import numpy as np
import torch

from pfl.stats import MappedVectorStatistics
from pfl.privacy import GaussianMechanism

# from utils.argument_parsing import parse_mechanism

torch.random.manual_seed(1)
np.random.seed(1)

# Define the Gaussian mechanism, use it to add noise to a vector of data, and show the mean and variance of the noise added

epsilon = 1
delta = 1e-8
cohort_size = 100
num_epochs = 1
population = cohort_size
clipping_bound = 2

gaussian_privacy = GaussianMechanism.construct_single_iteration(clipping_bound=clipping_bound, epsilon=epsilon, delta=delta)

# note that this can also be defined using the helper function
# gaussian_privacy = parse_mechanism(
#    mechanism_name='gaussian',
#    clipping_bound=clipping_bound,
#    epsilon=epsilon,
#    delta=delta)

# add the DP noise to 10000 random values
num_stats = 10000
statistics = MappedVectorStatistics({'stats': torch.randn(num_stats)})
noisy_statistics, metrics = gaussian_privacy.add_noise(statistics=statistics, cohort_size=1)

noise = noisy_statistics['stats'] - statistics['stats']

# report basic statistics about the added DP noise
noise_mean = torch.mean(noise)
noise_std_dev = torch.std(noise)
print(f'The empirical mean of the Gaussian noise added {num_stats} times = {noise_mean:.4f}')
print(f'The empirical standard deviation of the Gaussian noise added {num_stats} times = {noise_std_dev:.4f}')

# plot distribution of added DP noise
import matplotlib.pyplot as plt
plt.hist(noise, bins=60, density=True, label='Empirical distribution of Gaussian noise added for DP')

# Plot target Gaussian distribution of DP noise added: zero mean, std. dev. = clipping_bound * gaussian_privacy.relative_noise_stddev
gaussian_mean = 0
gaussian_std_dev = clipping_bound * gaussian_privacy.relative_noise_stddev
print(f'The true standard deviation of the noise generated using the Gaussian mechanism = {gaussian_std_dev:.4f}')
x = np.linspace(torch.min(noise), torch.max(noise), 1000)
plt.plot(x, norm.pdf(x, gaussian_mean, gaussian_std_dev), color='red', label='True distribution of Gaussian Mechanism for DP')

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title(f'Probability distribution of DP noise added {num_stats} times')
plt.show()
