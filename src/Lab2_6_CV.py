import numpy as np

def cross_validation(model, X, y, nFolds):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.

    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.

    Parameters:
    - model: scikit-learn-like estimator
        The machine learning model to be evaluated. This model must implement the .fit() and .score() methods
        similar to scikit-learn models.
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.

    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.
    """
    n_samples = X.shape[0]

    if nFolds == -1:
        nFolds = n_samples  # Leave-One-Out (LOO) CV

    fold_size = n_samples // nFolds
    accuracy_scores = []

    for i in range(nFolds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < nFolds - 1 else n_samples

        valid_indices = np.arange(start_idx, end_idx)
        train_indices = np.setdiff1d(np.arange(n_samples), valid_indices)

        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]

        model.fit(X_train, y_train)
        accuracy = model.score(X_valid, y_valid)
        accuracy_scores.append(accuracy)

    mean_score = np.mean(accuracy_scores)
    std_score = np.std(accuracy_scores)

    return mean_score, std_score

