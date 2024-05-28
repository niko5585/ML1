from sklearn.model_selection import train_test_split
from mlp_classifier_own import MLPClassifierOwn
import numpy as np


def train_nn_own(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifierOwn:
    """
    Train MLPClassifierOwn with PCA-projected features.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifierOwn object
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Create a MLPClassifierOwn object and fit it using (X_train, y_train)
    #       Print the train accuracy and validation accuracy
    #       Return the trained model
    clf = MLPClassifierOwn(5, 0.0, 32, (16,), 42)
    clf.fit(X_train, y_train)

    # train set
    train_accuracy = clf.score(X_train, y_train)
    # val test
    val_accuracy = clf.score(X_val, y_val)

    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    return clf
