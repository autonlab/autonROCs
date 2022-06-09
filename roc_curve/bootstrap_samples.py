# File: bootstrap_samples.py
# Author(s): Willa Potosnak
# Created: 12-15-21
# Last updated: 1-13-22
# Description: This code generates bootstrap samples

import numpy as np
import pandas as pd

def _bootstrap_samples(true_labels, predictions, bootstrap_iters=100, random_seed=0):
  """Generates bootstrap samples from classfier predictions

  Parameters
  -----------
  true_labels: pd.Dataframe
      A pandas dataframe containing the true labels
  predictions: pd.Dataframe
      A pandas dataframe containing the classifier predictions
  bootstrap_iters: integer, default=100
      The number of bootstrap iterations used to compute the ROC curve and confidence interval
  random_seed: integer, default=0
      Controls the randomness and reproducibility of the indices for generating bootstrap samples

  Returns
  --------
  bootstrap_truth: pd.Dataframe
      A pandas dataframe containing the true labels of bootstrap samples
  bootstrap_preds: pd.Dataframe
      A pandas dataframe containing the predictions of the classifier for boostrap samples
  folds: pd.Dataframe
      A pandas dataframe containing the iterations used for bootstrapping

    """
    
  bootstrap_truth = pd.DataFrame()
  bootstrap_predictions = pd.DataFrame()
  folds = pd.DataFrame()

  i = 0
  iter_num = 0
  while 1:
    np.random.seed(random_seed+i)
    i+=1
    resample_idxs = np.random.choice(range(len(true_labels)), len(true_labels))

    truth = true_labels.iloc[resample_idxs].reset_index(drop=True, inplace=False)
    preds = predictions.iloc[resample_idxs].reset_index(drop=True, inplace=False)

    if i>1000:
      raise Exception('Bootstrapping cannot proceed with provided sample class sizes')

    if set(truth.values.flatten().tolist()) == set(true_labels.values.flatten().tolist()):
      bootstrap_truth = pd.concat([bootstrap_truth, truth], axis=0)
      bootstrap_predictions = pd.concat([bootstrap_predictions, preds], axis=0)
      folds = folds.append(np.repeat(iter_num, len(true_labels)).tolist())
      iter_num += 1

      if iter_num == bootstrap_iters:
        folds.reset_index(drop=True, inplace=True)
        return bootstrap_truth, bootstrap_predictions, folds

    else:
      continue
