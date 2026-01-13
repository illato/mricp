import numpy as np

## 1. Hocevar T, Zupan B, Stålring J. Conformal Prediction with Orange. Journal of Statistical Software. 2021;98(7). doi:https://doi.org/10.18637/jss.v098.i07
##  => https://github.com/biolab/orange3-conformal

# nonconformity [1]
def inverse_probability(probs, target_idx):
    return 1 - probs[target_idx]
  
# nonconformity [1] (same as inverse_probability when binary classification)
def probability_margin(probs, target_idx):
    # prediciton probabilities for one example at a time
    assert probs.ndim == 1
    
    py = probs[target_idx]
    pz = max(prob for label_idx, prob in enumerate(probs) if label_idx != target_idx)
    return (1.0 - (py - pz)) / 2

# classification [1] (updated docstrings and removed dependencies)
class PredictionClass:
    """
    Conformal classification prediction object.

    This object encapsulates the results of a conformal classifier's prediction,
    including a list of candidate classes paired with their corresponding p-values.
    Each p-value represents the proportion of calibration predictions (computed either 
    from a pooled or class-specific calibration set) that had a prediction probability lower 
    than the current instance's predicted probability for that class. A higher p-value indicates 
    stronger support from the calibration data.

     Attributes:
        p (List[tuple]): A list of tuples (p_value, class) for each candidate class.
            Each p_value reflects the level of support for that candidate class.
        eps (float): The default significance level (error rate) used for forming prediction sets.
    """

    def __init__(self, p, eps):
        """
        Initialize the prediction.

        Args:
            p (List[tuple]): A list of tuples (p_value, class) for each candidate class.
                Each p_value is computed from calibration data and reflects the proportion
                of calibration predictions that had a lower predicted probability than 
                the current instance's prediction.
            eps (float): Default significance level (error rate).
        """
        self.p = p
        self.eps = eps

    def classes(self, eps=None):
        """
         Compute the set of candidate classes considered valid under the given significance level.

        Args:
            eps (float, optional): The significance level (error rate). If not provided, the default
                                     significance level (self.eps) is used.

        Returns:
            List: A list of classes for which the corresponding p_value is greater than eps.
        """
        if eps is None:
            eps = self.eps
            assert(eps is not None)
        return [y for p_y, y in self.p if p_y > eps]

    def verdict(self, ref, eps=None):
        """
        Determine if the prediction is correct based on the actual class.

        In conformal prediction, a prediction is considered correct if the true class is among
        the candidate classes computed for a given significance level.

        Args:
            ref: The reference/actual class.
            eps (float, optional): The significance level (error rate). Uses the default if not provided.

        Returns:
            bool: True if the reference/actual class appears in the prediction set, False otherwise.
        """
        return ref in self.classes(eps)

    def confidence(self):
        """
        Calculate the confidence of the prediction.

        Confidence reflects how clearly one class stands out (because the next best p‑value is much lower).

        Confidence is defined as 1 minus the second highest p_value among the candidate classes.
        In simpler terms, this measure indicates how distinctly the top prediction stands out compared 
        to the runner-up. A higher confidence means that the best candidate class is much better supported 
        than the next best alternative. 
        
        Note: This measure is not probabilistically sound--meaining it is not a reliable indicator of probabilistic
        quality (e.g., p-values→confidence: [0.95, 0.05]→0.95; [0.51, 0.49]→0.51; [0.30, 0.01]→0.99), which can be 
        misleading . However, it can still be useful (e.g., for confidence-based abstention), as it tells you 
        how much better the top prediction is compared to the rest.

        Computes minimum :math:`\\mathit{eps}` that would still result in a prediction of a single label.
        :math:`\\mathit{eps} = \\text{second\_largest}(p_i)`
        
        Returns:
            float: The confidence level, computed as 1 minus the second highest p_value.
                 :math:`1-\\mathit{eps}`
        """
        return 1-sorted([p_y for p_y, y in self.p], reverse=True)[1]

    def credibility(self):
        """
        Calculate the credibility of the prediction.

        A high credibility indicates that at least one class is highly plausible, whereas 
        a low credibility suggests the example is unusual.

        Credibility is defined as the highest p_value among all candidate classes.
        This measure represents the degree of support for the most plausible candidate class as derived 
        from the calibration data. A high credibility indicates strong support for at least one class,
        whereas a low credibility may suggest that even the best candidate is not very convincing.

        Computes minimum :math:`\\mathit{eps}` that would result in an empty prediction set.
        :math:`\\mathit{eps} = \\text{max}(p_i)`
        
        Returns:
            float: The credibility level, equal to the highest p_value.
                   :math:`\\mathit{eps}`
        """
        return max(p_y for p_y, y in self.p)

    def margin(self):
        """
        Calculate the margin between the conformal p-values of the two most supported classes.
        """
        return max(p_y for p_y, y in self.p) - sorted([p_y for p_y, y in self.p], reverse=True)[1]


def conformal_prediction(cal, test_in, alpha=0.1, class_conditional=False, verbose=True):
    """
    Generate conformal prediction sets directly using nonconformity scores with finite-sample correction.
    
    This function computes conformal prediction sets for a given test dataset based on a
    calibration dataset. It calculates p-values for each class prediction and forms prediction
    sets based on a specified significance level (alpha). Optionally, the computation can be done
    in a class-conditional manner.
    
    Parameters
    ----------
    cal : pd.DataFrame
        Calibration dataset representing classes {0, 1} with at least the following columns:
            - 'class': The true class label.
            - 'actual_class_pred_prob': The predicted probability for the true class.
    test : pd.DataFrame
        Test dataset containing predicted probabilities and the true class label. Expected columns:
            - 'pred_prob_0', 'pred_prob_1': Predicted probabilities for classes 0 and 1.
            - 'class': The true class label.
    alpha : float, optional (default=0.1)
        The significance level for conformal prediction. It is used as the epsilon threshold.
    class_conditional : bool, optional (default=False)
        If True, alphas are computed separately for each class using only calibration examples
        that belong to that class. If False, a global set of alphas is computed from all calibration examples.
    verbose : bool, optional (default=True)
        If True, prints empirical coverage and mean prediction set size. If False, suppresses print statements.

    
    Returns
    -------
    pd.DataFrame
        A copy of the input test DataFrame augmented with additional columns:
            - 'confidence': Confidence of the conformal prediction.
            - 'credibility': Credibility of the conformal prediction.
            - 'margin': Margin between conformal p-values of the two most supported classes.
            - 'classes': The prediction set (list of classes) from the conformal predictor.
            - 'verdict': Indicator (e.g., binary flag) showing if the true class is contained in the prediction set.
            - 'class_conditional': Indicates if calibration nonconformity scores were stratified by class in forming prediction sets.
            - 'cp': The raw conformal prediction objects.
    
    Side Effects
    ------------
    Prints the empirical coverage and mean prediction set size:
        - Prints the overall empirical coverage and the empirical coverage for each class.
        - Prints the overall mean prediction set size and the mean prediction set size for each class.
    
    Notes
    -----
    This implementation assumes a binary classification problem (classes 0 and 1). It relies on a
    `PredictionClass` object that must implement the following methods:
        - confidence()
        - credibility()
        - classes()
        - verdict(true_class)
        
    Examples
    --------
    >>> updated_test = conformal_prediction((calibration_df, test_df, alpha=0.1, class_conditional=True)
    >>> # This will print the empirical coverage and return the test DataFrame with added prediction details.
    
    """
    # Fail if any class missing from calibration set (else class will be in all prediction sets)
    required_classes = {0, 1}
    present_classes = set(cal['class'].unique())
    missing = required_classes - present_classes
    assert not missing, f"Calibration set is missing class(es): {missing}"
    
    # Make a copy of test_in to not mutate input df
    test = test_in.copy()
    
    # Precompute alphas based on the class_conditional flag.
    if class_conditional:
        # For each class (0 and 1), compute alphas from only those calibration examples that belong to that class.
        class_alphas = {
            cls: np.array(1 - cal.loc[cal['class'] == cls, 'actual_class_pred_prob'])
            for cls in [0, 1]
        }
    else:
        # Use all calibration examples.
        global_alphas = np.array(1 - cal['actual_class_pred_prob'])
    
    # Convert test probabilities to a numpy array.
    preds = test[['pred_prob_0', 'pred_prob_1']].to_numpy()
    
    # Build prediction results with a nested list comprehension.
    prediction_results = [
        PredictionClass(
            [
                (
                    # Use the appropriate alpha array based on the condition.
                    (np.sum((class_alphas[cls] if class_conditional else global_alphas) >= (1 - pred)) + 1)
                    / ((len(class_alphas[cls]) if class_conditional else len(global_alphas)) + 1),
                    cls
                )
                for cls, pred in enumerate(example)
            ],
            eps=alpha
        )
        for example in preds
    ]
    
    # Assign the PredictionClass results and compute the other columns.
    test['confidence'] = [cp.confidence() for cp in prediction_results]
    test['credibility'] = [cp.credibility() for cp in prediction_results]
    test['margin'] = [cp.margin() for cp in prediction_results]
    test['classes'] = [cp.classes() for cp in prediction_results]
    test['verdict'] = [cp.verdict(cls) for cp, cls in zip(prediction_results, test['class'])]
    test['class_conditional'] = class_conditional
    test['cp'] = prediction_results


    # Calculate and print empirical coverage and avg prediction set size if verbose is enabled.
    if verbose:        
        print(f'class_conditional: {class_conditional}')
        empirical_coverage = test['verdict'].mean()
        print(f"The empirical coverage is: {100 * empirical_coverage:.2f}%")
        
        # if class_conditional:
        # Compute empirical coverage per class.
        coverage_by_class = test.groupby('class')['verdict'].mean()
        for cls, coverage in coverage_by_class.items():
            print(f"Empirical coverage for class {cls}: {100 * coverage:.2f}%")
    
        pred_set_size = test['classes'].apply(len).mean()
        print(f"Mean prediction set size is: {pred_set_size:.2f}")
        
        # Compute mean prediction set size per class.
        pred_set_size_by_class = test.groupby('class')['classes'].apply(
            lambda x: x.map(len).mean()
        )
        for cls, ps_size in pred_set_size_by_class.items():
            print(f"Mean prediction set size for class {cls}: {ps_size:.2f}")
        print()
    
    return test

def conformal_prediction_quantile_based(cal, test, alpha=0.1, verbose=True):
    # set desired coverage
    # alpha = 0.1 # 1-alpha is the desired coverage
    
    # get conformal scores
    cal_scores = cal['actual_class_pred_prob']
    # get the adjusted quantile
    n = cal_scores.shape[0]
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(cal_scores, q_level, method='higher')
        
    # form conformal prediction sets
    pred_sets = test[['pred_prob_0', 'pred_prob_1']] >= (1 - qhat)
        
    # Calculate empirical coverage
    empirical_coverage = pred_sets.to_numpy()[np.arange(pred_sets.shape[0]),test['class']].mean()

    if verbose:
        print(f'\nqhat: {qhat}\n')
        print(pred_sets.apply(lambda x: int(x['pred_prob_0']) + int(x['pred_prob_1']), axis=1).value_counts())
        print(f"\nThe empirical coverage is: {empirical_coverage:.3f}\n\n")
