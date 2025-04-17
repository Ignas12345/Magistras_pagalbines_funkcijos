a = b
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
    return best_feature_set, performance_history