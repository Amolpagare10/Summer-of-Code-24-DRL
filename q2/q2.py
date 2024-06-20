import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None   
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    standardized_matrix = (init_array - init_array.mean()) #1

    covariance_matrix = np.cov(standardized_matrix.T) #2

    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix) #3

    sorted_indices = np.argsort(eigen_values)[::-1] #4
    sorted_eigenvalues_nr = eigen_values[sorted_indices]
    sorted_eigenvalues = np.round(sorted_eigenvalues_nr, 4)
    sorted_eigenvectors = eigen_vectors[:, sorted_indices]

    selected_eigenvectors = sorted_eigenvectors[:, :dimensions] #5

    final_data = np.dot(standardized_matrix, selected_eigenvectors) #6

    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("q2/pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(final_data[:, 0], final_data[:, 1], alpha=0.7)
    plt.title('PCA Transformed Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # plt.axis('equal')

    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.gca().set_aspect('equal', adjustable='box')


    plt.savefig('out.png')
    plt.show()

    # END TODO
