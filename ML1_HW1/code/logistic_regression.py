import numpy as np


def create_design_matrix_dataset_1(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 1.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    designMatrix = X_data
    thresholdValue_array = []

    for i in range(len(designMatrix)):
            if ((10 <= designMatrix[i][0] <30) and (0 <= designMatrix[i][1] <= 20 )): 
                thresholdValue_array.append(1)
            else:
                 thresholdValue_array.append(0)
    
    thresholdValue_array = np.array(thresholdValue_array).reshape(-1, 1)
    designMatrix = np.append(designMatrix, thresholdValue_array, axis=1)
    print(designMatrix)

    assert designMatrix.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert designMatrix.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return designMatrix

def create_design_matrix_dataset_2(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 2.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 2
    
    designMatrix = X_data

    thresholdValue_array = []
    euclidean_distance_array = []
    sq_distance_array = []


    start_x = 0
    start_y = 0
    r = 24
    r_sq = r ** 2

    def calculate_euclidean_distance(input_x, input_y):
        return (np.sqrt((input_x - start_x) ** 2 + (input_y - start_y) ** 2))

    def calculate_sq_distance(input_x, input_y):
        return ((input_x - start_x) ** 2 + (input_y - start_y) ** 2)

    for i in range(len(designMatrix)):
            euclidean_distance = calculate_euclidean_distance(designMatrix[i][0], designMatrix[i][1])
            sq_distance = calculate_sq_distance(designMatrix[i][0], designMatrix[i][1])

            euclidean_distance_array.append(euclidean_distance)
            sq_distance_array.append(sq_distance)

            if (((euclidean_distance) <= float(r)) and (designMatrix[i][0] <= r) and (designMatrix[i][1] <= r))\
                    and (sq_distance <= r_sq):
                thresholdValue_array.append(1)
            else:
                thresholdValue_array.append(0)
    
    thresholdValue_array = np.array(thresholdValue_array).reshape(-1, 1)
    euclidean_distance_array = np.array(euclidean_distance_array).reshape(-1, 1)
    sq_distance_array = np.array(sq_distance_array).reshape(-1, 1)

    designMatrix = np.append(designMatrix, thresholdValue_array, axis=1)
    designMatrix = np.append(designMatrix, euclidean_distance_array, axis=1)
    designMatrix = np.append(designMatrix, sq_distance_array, axis=1)

    print(designMatrix)

    assert designMatrix.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert designMatrix.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return designMatrix


def create_design_matrix_dataset_3(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 3.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 3
    designMatrix = X_data
    x1_power_2 = []
    x1_power_3 = []
    x1_power_4 = []

    x2_power_2 = []
    x2_power_3 = []
    x2_power_4 = []

    product = []
    sum = []


    for i in range(len(designMatrix)):
            x1_power_2.append(designMatrix[i][0] ** 2)
            x1_power_3.append(designMatrix[i][0] ** 3)
            x1_power_4.append(designMatrix[i][0] ** 4)

            x2_power_2.append(designMatrix[i][1] ** 2)
            x2_power_3.append(designMatrix[i][1] ** 3)
            x2_power_4.append(designMatrix[i][1] ** 4)

            product.append(designMatrix[i][0] * designMatrix[i][1])
            sum.append(designMatrix[i][0] + designMatrix[i][1])

    x1_power_2 = np.array(x1_power_2).reshape(-1, 1)
    x1_power_3 = np.array(x1_power_3).reshape(-1, 1)
    x1_power_4 = np.array(x1_power_4).reshape(-1, 1)
    x2_power_2 = np.array(x2_power_2).reshape(-1, 1)
    x2_power_3 = np.array(x2_power_3).reshape(-1, 1)
    x2_power_4 = np.array(x2_power_4).reshape(-1, 1)
    product = np.array(product).reshape(-1, 1)

    designMatrix = np.append(designMatrix, x1_power_2, axis=1)
    designMatrix = np.append(designMatrix, x1_power_3, axis=1)
    designMatrix = np.append(designMatrix, x1_power_4, axis=1)

    designMatrix = np.append(designMatrix, x2_power_2, axis=1)
    designMatrix = np.append(designMatrix, x2_power_3, axis=1)
    designMatrix = np.append(designMatrix, x2_power_4, axis=1)

    designMatrix = np.append(designMatrix, product, axis=1)


    assert designMatrix.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert designMatrix.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return designMatrix


def logistic_regression_params_sklearn():
    """
    :return: Return a dictionary with the parameters to be used in the LogisticRegression model from sklearn.
    Read the docs at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # TODO: Try different `penalty` parameters for the LogisticRegression model
    return {'penalty': 'l1', 'max_iter': 1000, 'solver': 'liblinear', 'multi_class': "ovr"}
