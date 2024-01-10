import random
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.metrics import recall_score

import random

def TT_Split(DF, t=.75):
    """
    Splits the DataFrame into training and testing sets based on unique dates.
    Training set includes a specified percentage of unique dates, and the rest are used for testing.

    Args:
        DF (pandas.DataFrame): The DataFrame to split.
        t (float, optional): The proportion of the dataset to include in the train split.

    Returns:
        tuple: Four DataFrames - train features, test features, train labels, test labels.
    """
    # Extract unique dates and shuffle them using random module k is the number of samples
    unique_dates = DF['date'].unique()
    shuffled_dates = random.sample(unique_dates, k=len(unique_dates))

    # Determine split index
    split_index = round(len(shuffled_dates) * t)

    # Split DataFrame into training and testing sets by placing into new data frames
    TrainDF = DF.loc[DF['date'].isin(shuffled_dates[:split_index])]
    TestDF = DF.loc[DF['date'].isin(shuffled_dates[split_index:])]

    # Extract labels from the DataFrames by placing into new data frames
    TrainL = TrainDF['label']
    TestL = TestDF['label']

    # Return features and labels, dropping unnecessary columns
    return (TrainDF.drop(['label', 'date', 'concentration'], axis=1),
            TestDF.drop(['label', 'date', 'concentration'], axis=1),
            TrainL, TestL)


def RFC_GridSearch(data):

    data_df = pd.concat(data, axis=1)
    # Split data into train and test sets
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .70)
    print(train_labels.shape, train_features.shape, test_labels.shape, test_features.shape)

    n_estimators = [10]
    max_features = ["sqrt", "log2"]
    max_depth = list(np.arange(10, 120, step=10))
    max_leaf_nodes = list(np.arange(10,510, step=50))
    min_samples_split = [2,4,6,8]
    min_samples_leaf = [1,2,3,4,8]
    max_samples=[.1,.2,.3,.4,.5,.6,.7,.8]
    bootstrap = [True]
    ###### GRID SEARCH ########

    param_grid = {
        "n_estimators":n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        'max_leaf_nodes':max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "max_samples": max_samples,
        "bootstrap":bootstrap}



    #acc_scorer = make_scorer(accuracy_score)
    print("beginning grid search")
    GRID_cv = HalvingGridSearchCV(RandomForestClassifier(),  param_grid, scoring='accuracy', cv=5, n_jobs=7, min_resources=14, error_score='raise', verbose=1)
    GRID_cv.fit(train_features, train_labels)
    gbp=GRID_cv.best_params_
    gbs=GRID_cv.best_score_

    rfc = RandomForestClassifier(n_estimators=1000, min_samples_split=gbp['min_samples_split'],
                                 min_samples_leaf=gbp['min_samples_leaf'],max_features=gbp['max_features'],
                                 max_depth=gbp['max_depth'],bootstrap=gbp['bootstrap'],
                                 max_leaf_nodes=gbp['max_leaf_nodes'],
                                 max_samples=gbp['max_samples'], oob_score=True, n_jobs=7)

    print('Classifier Parameters Found', ": ", rfc)
    return rfc, gbp, gbs

def RF_Testing(data, concentration, odors, classifier, P):
    """
    Trains a Random Forest classifier on the input data and returns accuracy score, confusion matrix,
    predictions, probabilities, predicted classes and log probabilities.

    Args:
    - data (pandas.DataFrame): Input data
    - concentration (str): Concentration to use for analysis
    - odors (str): Odor label to use for analysis
    - classifier (sklearn.ensemble.RandomForestClassifier): Random Forest classifier object

    Returns:
    - results_dict (dict): A dictionary containing the following keys:
      - 'accuracy_score': Accuracy score
      - 'confusion_matrix': Confusion matrix
      - 'predictions': Predictions
      - 'probabilities': Probabilities
      - 'predicted_classes': Predicted classes
      - 'log_probabilities': Log probabilities
    """

    # Concatenate the input data
    Analysis_data=pd.concat(data)

    # Get the data for the given concentration and odor label
    title=concentration
    data_df=Analysis_data[Analysis_data['concentration'].str.contains(title)]
    data_df=data_df[data_df['label'].str.contains(odors)]

    # Split the data into training and testing sets
    print('splitting data')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .7)

    print(classifier.get_params(deep=True))
    params = classifier.get_params()
    params['n_estimators'] = 1000
    classifier = RandomForestClassifier(**params)
    # Train the model on training data
    classifier.fit(train_features, train_labels)

    # Get predictions and probabilities on the testing set
    predictions = classifier.predict(test_features)
    probabilities = classifier.predict_proba(test_features)
    logprob= classifier.predict_log_proba(test_features)

    # Get the confusion matrix and accuracy score
    CM=confusion_matrix(test_labels,predictions, labels=classifier.classes_).astype(float)
    acc_score=balanced_accuracy_score(predictions, test_labels)
    r_score = recall_score(test_labels, predictions, zero_division=0, average='macro', labels=[P])
    print('Accuracy: ', acc_score, '\n')

    # Create a dictionary of results
    results_dict = {'classifier':classifier,
                    'accuracy_score': acc_score,
                    'sensitivity': r_score,
                    'confusion_matrix': CM,
                    'true classes':test_labels,
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'predicted_classes': classifier.classes_,
                    'log_probabilities': logprob}
    return results_dict

def SVM_GridSearch(data):
    """
    Perform a grid search to optimize SVM hyperparameters.

    Args:
    - data (List[pd.DataFrame]): A list of pandas dataframes, each containing the data to be analyzed
    - concentration (str): The concentration of the odor stimuli to be analyzed
    - odors (str): The label of the odor stimuli to be analyzed
    - P (str): The positive class label for computing recall score

    Returns:
    - clf (svm.SVC): The optimized SVM classifier
    - gbp (Dict[str, Any]): The best set of hyperparameters found by grid search
    - gbs (float): The best score found by grid search
    """

    # Concatenate data into a single dataframe
    #Analysis_data = pd.concat(data, axis=1)

    # Filter data based on concentration and odor label
    #data_df = Analysis_data[(Analysis_data['concentration'].str.contains(concentration)) &
                            #(Analysis_data['label'].str.contains(odors))]
    data_df = pd.concat(data, axis=1)
    # Split data into train and test sets
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .75)
    print(train_labels.shape, train_features.shape, test_labels.shape, test_features.shape)

    # Set hyperparameters to search over
    kernel = ['rbf']
    C = [0.5, 1, 4, 5, 7, 7.5, 8, 9, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 40]
    degree = [0, 0.01, 0.05, 0.1, 0.5]
    gamma = ['scale', 'auto', 0.1, 0.2, 0.5]
    coef0 = [0, 0.05, 0.1, 0.2]

    # Create parameter grid
    param_grid = {
        "kernel": kernel,
        "C": C,
        "degree": degree,
        "gamma": gamma,
        "coef0": coef0
    }

    print("Beginning grid search...")
    # Perform grid search
    GRID_cv = GridSearchCV(
        svm.SVC(),
        param_grid,
        scoring='accuracy',
        n_jobs=-1,
        error_score='raise',
        cv=15,
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

def SVM_Testing(data, concentration, odors, classifier, P):
    """
    Applies SVM classification on the provided data.

    Args:
    - data: pandas.DataFrame, input dataset
    - concentration: str, the concentration of odorant to classify
    - odors: str, the odorants to classify
    - classifier: sklearn.svm.SVC, the classifier to use
    - P: str, the positive class

    Returns:
    - results_dict: dict, a dictionary containing the results of the classification
    """

    # Filter the dataset based on concentration and odorants
    # Concatenate the input data
    Analysis_data=pd.concat(data)

    # Get the data for the given concentration and odor label
    title=concentration
    data_df=Analysis_data[Analysis_data['concentration'].str.contains(title)]
    data_df=data_df[data_df['label'].str.contains(odors)]
    global train_labels
    global test_labels

    # Split the dataset into training and testing sets
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .70)

    # Train the classifier
    print('Training model...')
    classifier.fit(train_features, train_labels)

    # Predict the response for test dataset
    print('Predicting classes...')
    predictions = classifier.predict(test_features)

    # Calculate accuracy and recall
    acc_score = balanced_accuracy_score(predictions, test_labels)
    r_score = recall_score(test_labels, predictions, zero_division=0, average='macro', labels=[P])

    # Compute the confusion matrix
    CM = confusion_matrix(test_labels, predictions, labels=classifier.classes_).astype(float)

    # Compute the decision function values
    probabilities = classifier.decision_function(test_features)

    # Print the results
    print('Accuracy: ', acc_score, '\n')
    print('Sensitivity: ', r_score, '\n')

    # Create a dictionary containing the results
    results_dict = {'classifier': classifier,
                    'accuracy_score': acc_score,
                    'sensitivity': r_score,
                    'confusion_matrix': CM,
                    'true classes': test_labels,
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'predicted_classes': classifier.classes_}

    return results_dict

def RF_Model_Test(concentrations, data, odor, PosL, repeats):
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
    for conc in [concentrations]:
        print(f"Beginning analysis for {odor} at {conc} concentration")
        classifier, params, best_score = RFC_GridSearch(data=data, concentration=conc, odors=odor)
        print(f"Best classifier for {odor} at {conc} concentration is {classifier}")
        print(f"Building Random Forest for {odor} at {conc} concentration")
        results.append([RF_Testing(data=data, classifier=classifier, concentration=conc, odors=odor, P=PosL)
                        for _ in range(repeats)])
        print(f"Finished analysis for {odor} at {conc} concentration")
    return results

def SVM_model_Test(concentrations, data, odor, PosL, repeats):
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
    for conc in [concentrations]:
        print(f"Beginning analysis for {odor} at {conc} concentration")
        classifier, params, best_score = SVM_GridSearch(data=data, concentration=conc, odors=odor)
        print(f"Best classifier for {odor} at {conc} concentration is {classifier}")
        print(f"Building SVM for {odor} at {conc} concentration")
        results.append([SVM_Testing(data=data, classifier=classifier, concentration=conc, odors=odor, P=PosL)
                        for _ in range(repeats)])
        print(f"Finished analysis for {odor} at {conc} concentration")
    return results

def pickle_Saver(savedir,ext,data):
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    reader=open(f'{savedir}{ext}.pickle','wb')
    pickle.dump(obj=data,file=reader)
    reader.close()



