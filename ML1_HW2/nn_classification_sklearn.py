from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings

# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # TODO: Create a PCA object and fit it using X_train
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_train)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance: {explained_variance:.2f}%")
    return X_reduced, pca

    # Transform X_train using the PCA object.
    # Print the explained variance ratio of the PCA object.
    # Return both the transformed data and the PCA object.


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Train MLPClassifier with different number of neurons in one hidden layer.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.

    num_hidden = [2, 10, 100, 200, 500]

    best_clf = None
    best_validation_accuracy = 0
    for n_hidden in num_hidden:
        mlp_classifier = MLPClassifier(max_iter=500, solver='adam', random_state=1, hidden_layer_sizes=(n_hidden,))
        mlp_classifier.fit(X_train, y_train)
        y_train_prediction = mlp_classifier.predict(X_train)
        y_validation_prediction = mlp_classifier.predict(X_val)
        train_accuracy = np.mean(y_train == y_train_prediction)
        validation_accuracy = np.mean(y_val == y_validation_prediction)
        final_training_loss = mlp_classifier.loss_
        print(f"n_hidden: {n_hidden}")
        print(f"Train accuracy: {train_accuracy * 100:.4f}%")
        print(f"Validation accuracy: {validation_accuracy * 100:.4f}%")
        print(f"Final training loss: {final_training_loss * 100:.4f}%")
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_clf = mlp_classifier

    return best_clf


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.
 

    best_model = None
    best_validation_accuracy = 0.0

    # CASE A

    mlp_A = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, solver='adam',
                        random_state=1, alpha=0.1, early_stopping=False)
    mlp_A.fit(X_train, y_train)
    train_accuracy = mlp_A.score(X_train, y_train)
    validation_accuracy = mlp_A.score(X_val, y_val)
    training_loss = mlp_A.loss_

    print(f"Alpha: {mlp_A.alpha}, Early Stopping: {mlp_A.early_stopping}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
    print(f"Training Loss: {training_loss:.4f}\n")

    # Track the best model
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_model = mlp_A

     # CASE B

    mlp_B = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, solver='adam',
                        random_state=1, alpha=0.0, early_stopping=True)
    mlp_B.fit(X_train, y_train)
    train_accuracy = mlp_B.score(X_train, y_train)
    validation_accuracy = mlp_B.score(X_val, y_val)
    training_loss = mlp_B.loss_

    print(f"Alpha: {mlp_B.alpha}, Early Stopping: {mlp_B.early_stopping}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
    print(f"Training Loss: {training_loss:.4f}\n")

    # Track the best model
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_model = mlp_B
    
     # CASE C

    mlp_C = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, solver='adam',
                        random_state=1, alpha=0.1, early_stopping=True)
    mlp_C.fit(X_train, y_train)
    train_accuracy = mlp_C.score(X_train, y_train)
    validation_accuracy = mlp_C.score(X_val, y_val)
    training_loss = mlp_C.loss_

    print(f"Alpha: {mlp_C.alpha}, Early Stopping: {mlp_C.early_stopping}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
    print(f"Training Loss: {training_loss:.4f}\n")
    

    # Track the best model
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_model = mlp_C


    print("Found  best model: alpha[%f] | early_stopping[%b]", best_model.alpha, best_model.early_stopping)
    print("Acc:", best_validation_accuracy)
    return best_model


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.

    # plt.plot(nn.loss_curve_)
    # plt.title('Training Loss Curve')
    #  plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    #  plt.grid(True)
    # plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(nn.loss_curve_)
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.

    nn()
confusion_matrix

def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.

    parameters = {
    'alpha': [0.0, 0.1, 1.0],  
    'solver': ['lbfgs', 'adam'],  
    'hidden_layer_sizes': [(100,), (200,)] 
    }

    mlp = MLPClassifier(max_iter=100, random_state=42)

    grid_search = GridSearchCV(mlp, parameters, cv=5, verbose=4, refit=True)
    grid_search.fit(X_train, y_train)
    est = grid_search.best_estimator_

    print("best estimator: ", est)
    print("set_params: ", grid_search.set_params)

    return est