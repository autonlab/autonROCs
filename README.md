This code can be used to plot Receiver Operating Characteric (ROC) curves. See the example notebook for more information.

## General Requirements:
    - Sample true labels and classifier predictions must be in a dataframe where each row refers to a different sample. 
    - True labels must be binary ({0, 1} or {-1, 1}) with 1 corresponding to the positive class
    - True label dataframe(s), prediction dataframe(s), fold dataframe(s), legend label(s), and line color(s) input must be stored in a list. Each list item must correspond to different classifier results to plot.
     
## Directions:

```python
# Import data
truth = [classifier_1_truth, classifier_2_truth]
preds = [classifier_1_predictions, classifier_2_predictions]
folds = [classifier_1_folds, classifier_2_folds]
labels = ['classifier 1', 'classifier 2']
colors = ['blue', 'red']

# Plot ROC curve using bokeh
from roc_curve.plotting import ROC_Curve_bokeh

plt = ROC_Curve_bokeh(axis_label_size='22', axis_tick_size='16', legend_text_size='18')
plt.plot(true_labels=truth, predictions=preds, folds=folds, xrange=(0,1), yrange=(0,1), direction='TPRvsFPR', 
         x_scale='linear', line_width=4, line_dash='solid', line_color=colors, legend_label=labels, 
         legend_location='bottom_right', ci_method='parametric', alpha=0.05, bootstrap_iters=None, random_seed=0)

# Plot ROC curve using matplotlib
from roc_curve.plotting import ROC_Curve_matplotlib

plt = ROC_Curve_matplotlib(axis_label_size='22', axis_tick_size='16', legend_text_size='18')
plt.plot(true_labels=truth, predictions=preds, folds=folds, xrange=(0,1), yrange=(0,1), direction='TPRvsFPR', 
         x_scale='linear', line_width=4, line_dash='solid', line_color=colors, legend_label=labels, 
         legend_location='bottom_right', ci_method='parametric', alpha=0.05, bootstrap_iters=None, random_seed=0,
         save_dir=None)
```

## Confidence Interval Options:
### parametric
    - Can be used for classifier results obtained from k-fold cross-validation
    - The standard error is computed based on the user-defined confidence level (alpha) and the number of k-folds. A 't' distribution is assumed rather than a 'z' distribution as the latter typically requires 'n>30' as one of the normal distribution assumption requirements. A t-distribution approaches a z-distribution for a large number of samples.
    - Side Note: k-fold score sets can be joined into one set provided the data meets the assumption that there is no difference in the population TPR and FPR between the k-folds. 

### bootstrap (non-parametric)
    - Can be used for results where k-folds is None
    - A bootstrap confidence interval is computed by resampling the same number of sample predictions and true label values as the original data WITH replacement. The user-defined confidence level is used to identify the empirical quantile values for the confidence interval.
