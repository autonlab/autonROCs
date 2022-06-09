# File: ci_methods.py
# Author(s): Willa Potosnak
# Created: 12-15-21
# Last updated: 1-13-22
# Description: This code computes confidence intervals

import numpy as np
import pandas as pd
from scipy.stats import norm, t

def _compute_parametric_ci(aggregated_roc_curves, num_folds, alpha=0.05):
  """A parametric approach to compute a confidence interval based on the
  user-defined confidence level (alpha) and the number of k-folds.

  Parameters
  -----------
  aggregated_roc_curves: pd.DataFrame
      A pandas dataframe containing the mean and standard deviation of tpr values
      at each fpr value.
  num_folds: integer
      The number of unique folds used for cross-validation .
  alpha: integer, default=0.05
      Confidence level for the t-distibution at which to compute the confidence
      interval.

  Returns
  --------
  pd.DataFrame
      A dataframe containing the lower and upper confidence interval bounds.

  """
  mean = aggregated_roc_curves['tpr']['mean']
  sd = aggregated_roc_curves['tpr']['std']

  critical_value = abs(t.ppf((1-alpha/2), num_folds-1))
  scaler = np.sqrt(num_folds)

  lb = mean - critical_value*(sd/scaler)
  ub = mean + critical_value*(sd/scaler)
  ci = pd.DataFrame({'lower_bound': lb, 'upper_bound': ub})

  return ci

def _compute_bootstrap_ci(interpolated_roc_curves, alpha=0.05):
  """A nonparametric approach to compute a confidence interval based on the
  empirical quantiles of bootstrap samples.

  Parameters
  -----------
  interpolated_roc_curves: pd.DataFrame
      A dataframe of concatenated bootstrap dataframes containing the interpolated
      true positive rate values and false positive rate values used for interpolation.
  alpha: integer, default=0.05
      Confidence level used to determine the empirical confidence interval.

  Returns
  --------
  pd.DataFrame
      A dataframe containing the lower and upper confidence interval bounds.

  """

  lb = interpolated_roc_curves.groupby('fpr').quantile(alpha/2)['tpr']
  ub = interpolated_roc_curves.groupby('fpr').quantile(1-alpha/2)['tpr']
  ci = pd.DataFrame({'lower_bound': lb, 'upper_bound': ub})

  return ci
