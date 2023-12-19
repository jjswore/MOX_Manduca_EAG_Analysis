import random
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

import pickle
from sklearn.metrics import recall_score

def TT_Split(DF, t=.75):
    """
    Splits the DataFrame into training and testing sets based on an identifier which is the unique
    date and antennae information stored under 'date' column.
    Training set includes a specified percentage of unique dates, and the rest are used for testing.

    Args:
        DF (pandas.DataFrame): The DataFrame to split.
        t (float, optional): The proportion of the dataset to include in the train split.

    Returns:
        tuple: Four DataFrames - train features, test features, train labels, test labels.
    """
    # Extract unique dates and shuffle them
    unique_dates = DF['date'].unique()
    shuffled_dates = random.sample(unique_dates, k=len(unique_dates))

    # Determine split index
    split_index = round(len(shuffled_dates) * t)

    # Split DataFrame into training and testing sets
    TrainDF = DF.loc[DF['date'].isin(shuffled_dates[:split_index])]
    TestDF = DF.loc[DF['date'].isin(shuffled_dates[split_index:])]

    # Extract labels from the DataFrames
    TrainL = TrainDF['label']
    TestL = TestDF['label']

    # Return features and labels, dropping unnecessary columns
    return (TrainDF.drop(['label', 'date', 'concentration'], axis=1),
            TestDF.drop(['label', 'date', 'concentration'], axis=1),
            TrainL, TestL)

def RFC_GridSearch(data):
    """
    Performs grid search to find optimal hyperparameters for RandomForestClassifier using the provided data.

    Args:
        data (list of pandas.DataFrame): The data to be used for grid search.

    Returns:
        tuple: The best-fitted RandomForestClassifier, best parameters, and best score.
    """
    # Concatenate provided data into a DataFrame
    data_df = pd.concat(data, axis=1)

    # Split data into train and test sets labels are the classes
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .70)

    # Define hyperparameters to search over
    param_grid = {
        "n_estimators": [10],  # Number of trees in the forest. we keep this small for the grid search
        "max_features": ["sqrt", "log2"],  # Number of features to consider when looking for the best split.
        "max_depth": list(np.arange(10, 120, step=10)),  # Maximum depth of the tree.
        'max_leaf_nodes': list(np.arange(10, 510, step=50)),  # Max number of leaf nodes in the tree.
        "min_samples_split": [2, 4, 6, 8],  # Minimum number of samples required to split an internal node.
        "min_samples_leaf": [1, 2, 3, 4, 8],  # Minimum number of samples required to be at a leaf node.
        "max_samples": [.1, .2, .3, .4, .5, .6, .7, .8],  # If bootstrap is True, the fraction of samples to be used for training each tree.
        "bootstrap": [True]  # Whether bootstrap samples are used when building trees.
    }

    # Perform grid search with halving strategy
    GRID_cv = HalvingGridSearchCV(RandomForestClassifier(), param_grid, scoring='accuracy', cv=5, n_jobs=7, min_resources=14, error_score='raise', verbose=1)
    GRID_cv.fit(train_features, train_labels)

    # Extract best parameters and score
    gbp = GRID_cv.best_params_
    gbs = GRID_cv.best_score_

    # Configure RandomForestClassifier with best parameters
    rfc = RandomForestClassifier(n_estimators=1000, min_samples_split=gbp['min_samples_split'],
                                 min_samples_leaf=gbp['min_samples_leaf'], max_features=gbp['max_features'],
                                 max_depth=gbp['max_depth'], bootstrap=gbp['bootstrap'],
                                 max_leaf_nodes=gbp['max_leaf_nodes'], max_samples=gbp['max_samples'],
                                 oob_score=True, n_jobs=7)

    print('Classifier Parameters Found:', rfc)
    return rfc, gbp, gbs


def RF_Train_and_Fit(data, classifier, recall_class):
    """
    Trains a Random Forest classifier on the input data filtered by specific concentration and odors,
    and evaluates its performance, returning various metrics and model details.

    Args:
        data (pandas.DataFrame): The dataset used for training and testing the classifier.
        concentration (str): The concentration of the odorant to be considered for analysis.
        odors (str): The specific odors to be used for filtering the data.
        classifier (RandomForestClassifier): Pre-initialized Random Forest classifier object.
        P (str): The label representing the positive class in the dataset.

    Returns:
        dict: A dictionary containing various evaluation metrics and model details:
            - 'classifier': The trained Random Forest classifier object.
            - 'accuracy_score': The balanced accuracy score of the model on the test set.
            - 'sensitivity': The recall score (sensitivity) of the model for the positive class.
            - 'confusion_matrix': The confusion matrix of the model predictions.
            - 'true_classes': The actual labels of the test set.
            - 'predictions': The predicted labels of the test set by the model.
            - 'probabilities': The class probabilities of the test set predictions.
            - 'predicted_classes': The classes that the classifier is trained to predict.
            - 'log_probabilities': The logarithm of class probabilities of the test set predictions.
    """

    # Concatenate input data frames. this is done if so that you can include multiple data sources
    # as input data
    data_df = pd.concat(data)

    # Split the filtered data into training and testing sets
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .7)

    # Update classifier parameters and train the classifier
    print('Training classifier with updated parameters...')
    params = classifier.get_params()
    params['n_estimators'] = 1000  # Update the number of trees in the forest
    classifier = RandomForestClassifier(**params)
    classifier.fit(train_features, train_labels)

    # Predict on the test set
    predictions = classifier.predict(test_features)
    probabilities = classifier.predict_proba(test_features)
    log_probabilities = classifier.predict_log_proba(test_features)

    # Calculate performance metrics
    confusion_matrix_result = confusion_matrix(test_labels, predictions, labels=classifier.classes_).astype(float)
    accuracy_score = balanced_accuracy_score(test_labels, predictions)

    #if a divide by zero occurs it assigns a value of 0 for the "sensitivity"
    sensitivity = recall_score(test_labels, predictions, zero_division=0, average='macro', labels=[recall_class])
    print('Accuracy:', accuracy_score, '\n')

    # Compile results into a dictionary
    results_dict = {
        'classifier': classifier,
        'accuracy_score': accuracy_score,
        'sensitivity': sensitivity,
        'confusion_matrix': confusion_matrix_result,
        'true_classes': test_labels,
        'predictions': predictions,
        'probabilities': probabilities,
        'predicted_classes': classifier.classes_,
        'log_probabilities': log_probabilities
    }

    return results_dict


def SVM_GridSearch(data):
    """
    Perform a grid search to optimize SVM hyperparameters.

    Args:
    - data (List[pd.DataFrame]): A list of pandas dataframes, each containing the data to be analyzed.

    Returns:
    - clf (svm.SVC): The optimized SVM classifier.
    - gbp (Dict[str, Any]): The best set of hyperparameters found by grid search.
    - gbs (float): The best score found by grid search.
    """
    data_df = pd.concat(data, axis=1)

    # Split data into train and test sets
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .75)

    # Set hyperparameters to search over
    param_grid = {
        "kernel": ['rbf'],  # Specifies the kernel type to be used in the algorithm.
        "C": [0.5, 1, 4, 5, 7, 7.5, 8, 9, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 40],  # Regularization parameter. affects distance between points and hyperplane
        "degree": [0, 0.01, 0.05, 0.1, 0.5],
        # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
        "gamma": ['scale', 'auto', 0.1, 0.2, 0.5],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. affects how many points and used in determining support vectors
        "coef0": [0, 0.05, 0.1, 0.2]
        # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    }

    print("Beginning grid search...")
    # Perform grid search
    GRID_cv = GridSearchCV(
        svm.SVC(),
        param_grid,
        scoring='accuracy',
        n_jobs=-1,
        error_score='raise',
        cv=15, #cross validation. used to determine training effectiveness
        verbose=1
    )
    GRID_cv.fit(train_features, train_labels)

    # Extract best hyperparameters and score
    gbp = GRID_cv.best_params_
    gbs = GRID_cv.best_score_

    # Create optimized classifier
    clf = svm.SVC(
        kernel=gbp['kernel'],
        C=gbp['C'],
        degree=gbp['degree'],
        gamma=gbp['gamma'],
        coef0=gbp['coef0']
    )

    print(f"Best parameters found: {gbp}")
    print(f"Best score found: {gbs}")
    print(f"Optimized classifier: {clf}")

    return clf, gbp, gbs

def SVM_Train_and_Fit(data, classifier, recall_class):
    """
    Trains an SVM classifier on the specified dataset filtered by concentration and odors,
    and evaluates its performance on the test set. Uses global variables to maintain state
    across multiple function calls if needed.

    Args:
        data (pandas.DataFrame): The dataset used for training and testing the classifier.
        concentration (str): The concentration of the odorant to be considered for analysis.
        odors (str): The specific odors to be used for filtering the data.
        classifier (sklearn.svm.SVC): Pre-initialized SVM classifier object.
        P (str): The label representing the positive class in the dataset. Used as a reference for sensitivity/recall

    Returns:
        dict: A dictionary containing various evaluation metrics and model details:
            - 'classifier': The trained SVM classifier object.
            - 'accuracy_score': The balanced accuracy score of the model on the test set.
            - 'sensitivity': The recall score (sensitivity) of the model for the positive class.
            - 'confusion_matrix': The confusion matrix of the model predictions.
            - 'true_classes': The actual labels of the test set.
            - 'predictions': The predicted labels of the test set by the model.
            - 'probabilities': The decision function values of the test set predictions.
            - 'predicted_classes': The classes that the classifier is trained to predict.
    """
    #designate train and test labels for verification purposes
    global train_labels
    global test_labels

    # Concatenate input data frames
    data_df = pd.concat(data)

    # Split the filtered data into training and testing sets
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .70)

    # Train the classifier
    print('Training classifier...')
    classifier.fit(train_features, train_labels)

    # Predict on the test set
    predictions = classifier.predict(test_features)

    # Compute performance metrics
    accuracy_score = balanced_accuracy_score(test_labels, predictions)
    sensitivity = recall_score(test_labels, predictions, zero_division=0, average='macro', labels=[recall_class])
    confusion_matrix_result = confusion_matrix(test_labels, predictions, labels=classifier.classes_).astype(float)

    # Compute decision function values (probabilities)
    decision_function_values = classifier.decision_function(test_features)

    # Compile results into a dictionary
    results_dict = {
        'classifier': classifier,
        'accuracy_score': accuracy_score,
        'sensitivity': sensitivity,
        'confusion_matrix': confusion_matrix_result,
        'true_classes': test_labels,
        'predictions': predictions,
        'probabilities': decision_function_values,
        'predicted_classes': classifier.classes_
    }

    return results_dict

def RF_model_Testing(data, recall_class, repeats):
    """
    Trains and evaluates a Random Forest classifier on the provided dataset for each concentration of a given odor.

    Args:
        concentrations (list): A list of concentrations of the odor to be analyzed.
        data (pd.DataFrame): A pandas DataFrame containing the training data.
        odor (str): The name of the odor being analyzed.

    Returns:
        list: A nested list containing the classification performance metrics for each iteration of the Random Forest classifier
            for each concentration of the given odor.
    """
    results = []

    print(f"Beginning Random Forest hyperparameter optimization")
    classifier, params, best_score = RFC_GridSearch(data=data)
    print(f"Best classifier found. Classifier hyperparemeters include: {classifier}")
    print(f"Testing splits of data for random forest model")
    results.append([RF_Train_and_Fit(data=data, classifier=classifier, recall_class=recall_class)
                    for _ in range(repeats)])
    print(f"Finished Model Testing of Random Forest")
    return results

def SVM_model_Testing(data, recall_class, repeats):
    """
    Trains and evaluates a Support Vector Machine classifier on the provided dataset for each concentration of a given odor.

    Args:
        concentrations (list): A list of concentrations of the odor to be analyzed.
        data (pd.DataFrame): A pandas DataFrame containing the training data.
        odor (str): The name of the odor being analyzed.

    Returns:
        list: A nested list containing the classification performance metrics for each iteration of the Support Vector Machine classifier
            for each concentration of the given odor.
    """
    results = []
    print(f"Beginning SVM hyperparameter optimization")
    classifier, params, best_score = SVM_GridSearch(data=data)
    print(f"Best classifier found. Classifier hyperparemeters include: {classifier}")
    print(f"Testing splits of data for SVM model")
    results.append([SVM_Train_and_Fit(data=data, classifier=classifier, recall_class=recall_class)
                    for _ in range(repeats)])
    print(f"Finished Model Testing of SVM")
    return results

def pickle_Saver(savedir, ext, data):
    """
    Saves the provided data to a pickle file in the specified directory.

    Args:
        savedir (str): The directory where the pickle file will be saved.
        ext (str): The name or extension for the pickle file.
        data (any): The data to be saved in the pickle file.

    This function checks if the specified directory exists, and if not, it creates the directory.
    Then, it saves the provided data into a pickle file named as per the 'ext' argument within
    the specified 'savedir' directory. The data is written in binary format ('wb').
    """

    # Check if the directory exists, if not, create it
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # Open the file in write-binary mode and save the data
    with open(f'{savedir}{ext}.pickle', 'wb') as reader:
        pickle.dump(obj=data, file=reader)

    # Note: The file is automatically closed when exiting the 'with' block