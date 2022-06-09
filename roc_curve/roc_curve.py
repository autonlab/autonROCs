# File: roc_curve.py
# Author(s): Willa Potosnak
# Created: 12-15-21
# Last updated: 1-13-22
# Description: This code computes ROC curves

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .bootstrap_samples import _bootstrap_samples
from .ci_methods import _compute_parametric_ci, _compute_bootstrap_ci

def compute_roc_curve(true_labels, predictions, folds=None, thresholds=None,
                      ci_method='parametric', alpha=0.05, bootstrap_iters=100,
                      random_seed=0):

  """Computes the ROC curve values.

  Parameters
  -----------
  true_labels: pd.Dataframe
      A pandas dataframe containing the true labels.
  predictions: pd.Dataframe
      A pandas dataframe containing classifier predictions.
  folds: pd.Dataframe, default=None
      A pandas dataframe containing the fold numbers used for cross-validation.
  thresholds: pd.Dataframe or dict, default=None
      A dictionary or pandas dataframe containing false positive rates over
      which to interpolate true positive rate values.
  ci_method: string, default='parametric'
      Options include:'parametric' or 'bootstrap'
      The parametric method computes the standard error based on the number of folds and
      the user-defined significance level (alpha) for a t-distribution.
      The bootstrap method uses empirical quantiles from bootstrap samples generated
      by resampling the same number of sample scores and outcome values as the original
      data with replacement.
  alpha: integer, default=0.05
      Confidence level for the t-distibution at which to compute the confidence interval
  bootstrap_iters: integer, default=100
      The number of bootstrap iterations used to compute the ROC curve and confidence interval.
  random_seed: integer, default=0
      Controls the randomness and reproducibility of the indices for generating bootstrap samples

  Returns
  --------
  pd.DataFrame
      A dataframe containing the true positive rate and false postive rate values.

  """

  if ci_method == 'parametric':
    if folds is None:
      raise Exception("fold input is required to use the 'parametric' CI method")
    if bootstrap_iters is not None:
      raise Exception("bootstrap_iters input must be 'None' to use the 'parametric' CI method")
  elif ci_method == 'bootstrap':
    if folds is not None:
      raise Exception("fold input must be 'None' to use the 'bootstrap' CI method")
  else:
    raise Exception('CI method is not supported')

  if ci_method == 'bootstrap':
    true_labels, predictions, folds = _bootstrap_samples(true_labels, predictions, bootstrap_iters)
    unique_iters = set(folds.values.flatten())
  else:
    unique_iters = set(folds.values.flatten())

  fpr_interpolated_roc_curves = []
  fnr_interpolated_roc_curves = []
  for fold in unique_iters:
    fpr, tpr, fnr, tnr = _compute_roc_curve(true_labels[np.array(folds == fold)],
                                            predictions[np.array(folds == fold)])
    fpr_interpolated_roc_curves.append(_interpolate_roc_curves(fpr, tpr, thresholds))
    fnr_interpolated_roc_curves.append(_interpolate_roc_curves(fnr, tnr, thresholds))
        
  fpr_aggregated_roc_curve = _aggregate_roc_curves(fpr_interpolated_roc_curves)
  fnr_aggregated_roc_curve = _aggregate_roc_curves(fnr_interpolated_roc_curves)

  if ci_method == 'parametric':
    fpr_ci = _compute_parametric_ci(fpr_aggregated_roc_curve, len(unique_iters), alpha)
    fnr_ci = _compute_parametric_ci(fnr_aggregated_roc_curve, len(unique_iters), alpha)
  else:
    fpr_ci = _compute_bootstrap_ci(pd.concat(fpr_interpolated_roc_curves), alpha)
    fnr_ci = _compute_bootstrap_ci(pd.concat(fnr_interpolated_roc_curves), alpha)

  return fpr_aggregated_roc_curve, fpr_ci, fnr_aggregated_roc_curve, fnr_ci

def _interpolate_roc_curves(fpr, tpr, thresholds):
  """Interpolates tpr over fpr values or tnr over fnr values.

  Parameters
  -----------
  fpr: pd.Dataframe of false positive rate values
  tpr: pd.Dataframe of true positive rate values
  pd.Dataframe or dict, default=None
      A dictionary or pandas dataframe containing false positive rates over
      which to interpolate true positive rate values.

  Returns
  --------
  pd.DataFrame
      A dataframe containing the interpolated true positive rate values and
      user-defined false positive rate values used for interpolation.

  """

  if thresholds is None:
    thresholds=np.linspace(0, 1, 500)
    
  tpr_interp = np.interp(thresholds, fpr, tpr, left=0, right=1)
  interpolated_roc_curve = pd.DataFrame({'fpr': thresholds, 'tpr': tpr_interp})

  return interpolated_roc_curve

def _aggregate_roc_curves(interpolated_roc_curves):
  """Aggregate ROC curve results over folds or bootstrap iterations.

  Parameters
  -----------
  interpolated_roc_curves: pd.DataFrame
      A dataframe containing the interpolated true positive rate values and user-defined false positive rate values
      used for interpolation.

  Returns
  --------
  pd.DataFrame
      A dataframe containing the aggregated true positive rate mean and standard deviation indexed by the 
      user-defined false positive rate values used for interpolation.

  """

  agg_roc_curves = pd.concat(interpolated_roc_curves).groupby('fpr').agg([np.mean, np.std])
  agg_roc_curves.sort_index(ascending=True, inplace=True)

  return agg_roc_curves

def _compute_roc_curve(true_labels, predictions):
  """Computes fpr, tpr, fnr, tnr.

  Parameters
  -----------
  scores: pd.Dataframe or dict
      A dictionary or pandas dataframe containing the scores of the classifier and the true label.

  Returns
  --------
  fpr: pd.Dataframe of false positive rate values
  tpr: pd.Dataframe of true positive rate values
  fnr: pd.Dataframe of false negative rate values
  tnr: pd.Dataframe of true negative rate values

  """

  fpr, tpr, thres = roc_curve(true_labels.values.flatten(), predictions.values.flatten())
  fnr = (1-tpr)[::-1]
  tnr = (1-fpr)[::-1]

  return fpr, tpr, fnr, tnr
