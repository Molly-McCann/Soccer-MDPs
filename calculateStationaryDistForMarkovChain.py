import numpy as np

def calculate_stationary_distribution(transition_matrix):
    '''
    INPUTS
    transition_matrix: np.array of size nxn

    OUTPUTS

    '''
    # Ensure the transition matrix is a valid square matrix
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("Transition matrix must be a square matrix")

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    print(f' Eigenvalues of T matrix {eigenvalues}')
    
    print(f'Eigenvectors of T matrix {eigenvectors}')

    # Find the index of the eigenvalue 1 (or close to 1)
    index_of_unity_eigenvalue = np.where(np.isclose(eigenvalues, 1))[0][0]

    # print(eigenvectors[:, index_of_unity_eigenvalue])
    # Extract the corresponding eigenvector
    stationary_distribution = np.real(eigenvectors[:, index_of_unity_eigenvalue])
    
    # Normalize the stationary distribution
    stationary_distribution /= np.sum(stationary_distribution)

    return stationary_distribution

def main():
    # Example usage:
    # Replace the following matrix with your transition matrix
    # transition_matrix = np.array([[0.7, 0.2,0.1],[0.4, 0.6,0.0],[0,1,0]])
    transition_matrix = np.array([[0.2, 0.6,0.2],[0.8, 0.2,0.0],[0.4,0.4,0.2]])

    # transition_matrix = np.array([[0.0, 0.2,0.8],[0.0, 0.6,0.4],[0,0.5,0.5]])

    stationary_distribution = calculate_stationary_distribution(transition_matrix)

    print("Stationary Distribution:", stationary_distribution)



if __name__ == '__main__':
    main()
