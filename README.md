#Install pymc
!pip install pymc==4.1.4
import pymc as pm
pm.__version__

#Import numpy, panda and import your data set 
import numpy as np
import pandas as pd
from google.colab import files
import io
uploaded=files.upload()
data=pd.read_excel(io.BytesIO(uploaded['1data2.xlsx']))
data
data = data.dropna()
import matplotlib.pyplot as plt
import arviz as az
data['Index'] = pd.Categorical(data['Index'])

# Fixed effects for Lunar Cycle and Interest Rate, individual-specific effects (Index) and Model Equation
with pm.Model() as model_panel:
    beta_lunar = pm.Normal('beta_lunar', mu=0, sigma=10)
    beta_pinterest = pm.Normal('beta_pinterest', mu=0, sigma=10)
    beta_sinterest = pm.Normal('beta_sinterest', mu=0, sigma=10)
    beta_balance = pm.Normal('beta_balance', mu=0, sigma=10)
    alpha_index = pm.Normal('alpha_index', mu=0, sigma=10, shape=len(data['Index'].unique()))
    mu = alpha_index[data['Index'].cat.codes] + beta_lunar * data['X1'] + beta_sinterest * data['Secondary Rate'] + beta_pinterest * data['Primary Rate'] + beta_balance * data['Balance at NRB - CRR']

# Likelihood
sigma = pm.HalfNormal('sigma', sigma=1)
y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data['Close_Price'])
with model_panel:
 trace_panel = pm.sample(1000, tune=1000)
import arviz as az
import matplotlib.pyplot as plt
az.plot_trace(trace_panel)
plt.show()
print(pm.summary(trace_panel))

def calculate_cedes(chains):
    n_params = chains.shape[2]  # Assuming chains is a 3D array: (n_chains, n_samples, n_params)
    n_chains = chains.shape[0]
    n_samples = chains.shape[1]
# Calculate means and variances
chain_means = np.mean(chains, axis=1)
within_chain_variances = np.var(chains, axis=1)

# Calculate between-chain variance
between_chain_variance = np.var(chain_means, axis=0)
# Calculate pooled variance estimate
pooled_variance_estimate = ((n_samples - 1) / n_samples) * np.mean(within_chain_variances, axis=0) + (1 / n_samples) * between_chain_variance
# Calculate potential scale reduction factor (R-hat)
potential_scale_reduction_factor = np.sqrt(pooled_variance_estimate / np.mean(within_chain_variances, axis=0))
return potential_scale_reduction_factor
# Example usage
n_chains = 4
n_samples = 1000
n_params = 3
chains = np.random.normal(loc=0, scale=1, size=(n_chains, n_samples, n_params))
rhats = calculate_cedes(chains)
print("R-hat values:", rhats)
# Density plots
az.plot_posterior(trace_panel, var_names=['beta_lunar', 'beta_pinterest', 'beta_sinterest', 'beta_balance'])
plt.show()
# Autocorrelation plots
az.plot_autocorr(trace_panel, var_names=['beta_lunar', 'beta_pinterest', 'beta_sinterest', 'beta_balance'])
plt.show()
# Extract mean coefficient values
intercept_mean = 113.478
beta_lunar_mean = 113.478
beta_pinterest_mean = 237.563
beta_sinterest_mean = 173.141
beta_balance_mean = 0.015

# Formulate the regression equation
regression_equation = f"y = {intercept_mean:.2f} + {beta_lunar_mean:.2f} * Lunar + {beta_pinterest_mean:.2f} * Pinterest + {beta_sinterest_mean:.2f} * Secondary Interest + {beta_balance_mean:.2f} * Balance"

print("Regression equation:", regression_equation)
