import numpy as np 
import matplotlib.pyplot as plt


def create_adjacency_matrix(num_frames, mode = 'window', window_size = 5):
    """
    Create a custom adjacency matrix for the graph.
    Since all our MFCCs are of the same length, we can create a static adjacency matrix.
    The adjacency matrix is used to define the connections between nodes in the graph.
    The adjacency matrix is created based on the mode specified.
    
    Args:
        num_frames: Number of frames in the MFCC.
        mode: Sets the mode in which the adjacency matrix is created.
               'window' : Creates an adjacency matrix based on a sliding window over the frames.
                          This means that each frame is connected to its 'window_size' neighbors.
   
        
        Returns:
        adjacency_matrix: A 2D numpy array representing the adjacency matrix.
    """

    adjacency_matrix = np.zeros((num_frames, num_frames), dtype=np.float32)
    
    if mode == 'window':
        # Create a sliding window adjacency matrix based on the window size
        for i in range(num_frames):
            start = i
            end = min(num_frames, i + window_size)
            adjacency_matrix[i, start:end] = 1.0
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    
    return adjacency_matrix


def visualize_adjacency_matrix(adjacency_matrix, title="Adjacency Matrix"):
    """
    Visualize an adjacency matrix as a heatmap.
    
    Args:
        adjacency_matrix: 2D numpy array representing the adjacency matrix
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(adjacency_matrix, cmap='viridis')
    plt.colorbar(label='Connection weight')
    plt.title(title)
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    plt.tight_layout()
    plt.show()