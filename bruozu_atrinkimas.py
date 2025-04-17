import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from copy import deepcopy
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

#ChatGPT funkcija
def forward_feature_selection_cv(X_cv, y_cv, classifier, max_features=20, allow_n_rounds_without_improvement = 3, n_splits=5, seed=42):
    """
    Perform forward feature selection over the provided set of candidate features
    (which could be the top DE features) and evaluate the model using stratified k-fold
    cross-validation. The function returns the set of features that yielded the best
    validation performance.

    Parameters:
        X_cv (pd.DataFrame): The dataframe containing the candidate features (columns).
        y_cv (array-like): The labels corresponding to the rows in X_cv.
        classifier: The scikit-learn classifier to be used.
        max_features (int): Maximum number of features to include. The default is 20.
        n_splits (int): Number of folds for StratifiedKFold.
        seed (int): Random state seed for reproducibility.

    Returns:
        best_feature_set (list): List of feature names that produced the best CV score.
        performance_history (list of dicts): A log of performance at each round.
    """
    # Ensure the candidate features are in a list.
    candidate_features = list(X_cv.columns)
    current_features = []
    remaining_features = deepcopy(candidate_features)

    best_score_overall = 0
    best_feature_set = []
    performance_history = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print("Starting forward feature selection with cross-validation...")

    # Loop until we've reached the maximum allowed features or no more remain
    while remaining_features and (len(current_features) < max_features):
        best_score_this_round = 0
        best_feature_this_round = None

        # Evaluate each candidate feature by adding it to the current set.
        for feature in remaining_features:
            temp_features = current_features + [feature]
            X_sub = X_cv[temp_features]
            # Evaluate with cross-validation.
            scores = cross_val_score(classifier, X_sub, y_cv, cv=skf)
            mean_score = scores.mean()

            if mean_score > best_score_this_round:
                best_score_this_round = mean_score
                best_feature_this_round = feature

        if best_feature_this_round is None:
            # No improvement is possible.
            print('best_feature_this_round is None')
            break

        # Update the current set and remove the selected feature from remaining features.
        current_features.append(best_feature_this_round)
        remaining_features.remove(best_feature_this_round)
        performance_history.append({
            "num_features": len(current_features),
            "cv_score": best_score_this_round,
            "features": deepcopy(current_features)
        })
        print(f"Round {len(current_features)}: Added '{best_feature_this_round}' -> CV Score: {best_score_this_round:.4f}")

        # Update overall best if this round's score improved.
        if best_score_this_round > best_score_overall:
            best_score_overall = best_score_this_round
            best_feature_set = deepcopy(current_features)

        #add stopping criterion: if accuraccy does not improve in n rounds, stop process:
        #if len(performance_history) >= 3 and performance_history[-1]["cv_score"] <= performance_history[-2]["cv_score"] and performance_history[-1]["cv_score"] <= performance_history[-3]["cv_score"]:
        if len(performance_history) >= (allow_n_rounds_without_improvement+1) and performance_history[-1]["cv_score"] <= performance_history[-(allow_n_rounds_without_improvement+1)]["cv_score"]:
            break

    print("Forward feature selection completed!")
    print(f"Best feature set ({len(best_feature_set)} features) with CV score: {best_score_overall:.4f}")
    return best_feature_set

def sfs_feature_selection(X_cv, y_cv, classifier, n_features=20, tol = None, n_splits=5, seed=42):
    """
    Perform forward feature selection using scikit-learn's SequentialFeatureSelector.

    Parameters:
        X_cv (pd.DataFrame): DataFrame of candidate features.
        y_cv (array-like): Target class labels.
        classifier: A scikit-learn compatible classifier.
        n_features (int): Number of features to select (upper limit).
        n_splits (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.

    Returns:
        selected_features (list): List of feature names selected by SFS.
        cv_score (float): Mean cross-validation score with the selected features.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Create the SequentialFeatureSelector object; note that direction='forward' configures forward selection.
    sfs = SequentialFeatureSelector(
        classifier,
        n_features_to_select=n_features,
        tol = tol,
        direction='forward',
        cv=cv
    )

    # Fit the SFS model to perform feature selection.
    sfs.fit(X_cv, y_cv)

    # Get a boolean mask of the selected features.
    selected_mask = sfs.get_support()
    selected_features = X_cv.columns[selected_mask].tolist()

    # Evaluate the performance using cross-validation on the selected features.
    scores = cross_val_score(classifier, X_cv[selected_features], y_cv, cv=cv)
    cv_score = scores.mean()

    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Cross-Validation Score: {cv_score:.4f}")

    return selected_features


def kfold_cv(X_cv, y_cv, final_classifier, n_splits=5, print_incorrect = False, seed=42):
    """
    Perform stratified k-fold cross-validation for a given classifier.

    Parameters:
        X_cv (pd.DataFrame): Feature set.
        y_cv (pd.Series or np.array): Labels.
        final_classifier: A scikit-learn compatible classifier.
        label_dict (dict, optional): Optional mapping for labels.
        seed (int): Random seed for reproducibility.
        n_splits (int): Number of folds.

    Returns:
        results_df (pd.DataFrame): DataFrame containing sample names, correctness of prediction,
                                   and the fold number.
    """
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    total_misclassified = 0
    print("\nStarting Stratified K-Fold Cross-Validation\n")
    # Loop through the folds; using tqdm for a progress bar with total equal to n_splits
    for fold, (train_index, test_index) in enumerate(tqdm(skf.split(X_cv, y_cv), total=n_splits, desc="CV Progress"), start=1):
        # Split data into training and testing sets based on the current fold
        X_train, X_test = X_cv.iloc[train_index], X_cv.iloc[test_index]
        y_train, y_test = y_cv[train_index], y_cv[test_index]

        # Train the classifier on the training set
        final_classifier.fit(X_train, y_train)
        # Predict on the test set
        y_pred = final_classifier.predict(X_test)

        # Iterate over the test set samples
        for sample_idx, pred, true in zip(test_index, y_pred, y_test):
            sample_name = X_cv.index[sample_idx]  # Get the sample name from the index
            results.append({
                "Sample": sample_name,
                "Correct": int(pred == true),
                "Fold": fold
            })
        #print accuracy on test fold i, incorrect samples in that fold
        incorrect_samples = X_cv.iloc[test_index][y_pred != y_test].index.to_list()
        total_misclassified += len(incorrect_samples)
        print(f"Fold {fold} accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Number of misclassified samples in Fold {fold}: {len(incorrect_samples)}")
        if print_incorrect and len(incorrect_samples) > 0:
          print(f"Incorrectly classified samples in Fold {fold}:")
          print(incorrect_samples)

    print("\nCross-Validation completed!")
    # Convert the list of results to a DataFrame for further analysis
    results_df = pd.DataFrame(results)
    # Print summary
    print("\nOverall Accuracy:", results_df["Correct"].mean())
    print("total misclassified samples: ")
    print("\nDetailed Results:")

    return results_df

def loo_cv(X_cv, y_cv, final_classifier, label_dict = None, seed = 42):

  np.random.seed(seed)
  loo = LeaveOneOut()
  results = []

  # Progress bar for tracking LOOCV progress
  print("\nStarting Leave-One-Out Cross-Validation\n")
  for train_index, test_index in tqdm(loo.split(X_cv), total=np.shape(X_cv)[1], desc="LOOCV Progress"):
      # Split data
      X_train, X_test = X_cv.iloc[train_index], X_cv.iloc[test_index]
      y_train, y_test = y_cv[train_index], y_cv[test_index]
      test_sample_name = X_cv.index[test_index][0]  # Get left-out sample name


      # Train model with optimal features
      #svc_final = SVC(kernel="linear", C = 0.1, class_weight = 'balanced', random_state=42, probability=True)
      final_classifier.fit(X_train, y_train)

      # Predict class and get confidence
      y_pred = final_classifier.predict(X_test)
      #decision_values = svc_final.decision_function(X_test)  # Signed distance to the decision boundary
      #confidence_score = np.abs(decision_values[0])  # Larger magnitude → higher confidence

      # Compute margin width
      #support_vectors = svc_final.support_vectors_
      #dual_coef = svc_final.dual_coef_
      #margin_width = 2 / np.linalg.norm(dual_coef)  # Width of the margin

      # Store results
      results.append({
          "Sample": test_sample_name,
          "Correct": int(y_pred[0] == y_test[0]),
          #"Confidence": confidence_score,
          #"Margin Width": margin_width,
      })

  print("\nLOOCV completed!")

  # Convert results into a DataFrame
  results_df = pd.DataFrame(results)

  # Print summary
  print("\nOverall LOOCV Accuracy:", results_df["Correct"].mean())
  print("\nDetailed Results:")
  print(results_df)

  incorrect_samples = results_df[results_df['Correct'] == 0]
  print("\nIncorrectly classified samples:")
  print(incorrect_samples)
  if label_dict is not None:
      for sample in incorrect_samples['Sample']:
          print(sample + ' label is: ' + label_dict[sample] + '\n')

  return results_df