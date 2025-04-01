import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def create_adjacency_matrix(mfcc, num_frames, label, mode = 'window', window_size = 5, alpha = 0.6, beta = 0.2):
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

    adjacency_matrix = tf.zeros((num_frames, num_frames), dtype=np.float32)
    
    if mode == 'window':
        # Create a sliding window adjacency matrix based on the window size

        indices = tf.range(num_frames)

        # Create a column & row vector of the indices
        i = tf.reshape(indices, [-1, 1])
        j = tf.reshape(indices, [1,-1])

        # Calculate the absolute distance between numbers ; creates a matrix of distances
        # in size of (num_frames, num_frames)
        distance = tf.abs(i-j)

        # Based on that distance, create the adjacency matrix (by casting to 1 or 0)
        adjacency_matrix = tf.cast(distance <= window_size, dtype=tf.float32)



    elif mode == 'similarity':
        # Create a similarity adjacency matrix based on the cosine similarity between frames, with a penalty for distance
        # This should ideally cluster close frames with similar frequencies together, allowing for an identification of the phonemes/words
        adjacency_matrix = similarity_function(mfcc, num_frames, alpha=alpha, beta=beta)

    
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    
    # Fix self loops by setting diagonal to 0
    adjacency_matrix = tf.linalg.set_diag(adjacency_matrix, tf.zeros(num_frames, dtype=tf.float32))


    return (mfcc, adjacency_matrix) , label



#TODO: alpha and beta are best to be tuned on a validation set
def similarity_function(mfccs, num_frames, alpha, beta):

    # Take as input the MFCCs of a single audio file (not batched)
    # Initialize
    feature_similarity = tf.zeros((num_frames, num_frames))
    temporal_similarity = tf.zeros((num_frames, num_frames))
    similarity_matrix = tf.zeros((num_frames, num_frames))

    # 1. Feature_similarity:
    # Normalize the feature vectors for cosine similarity
    normalized_features = tf.nn.l2_normalize(mfccs, axis=1)

    # Compute the normalized cosine similarity between all pairs of frames
    for i in range(num_frames):
        for j in range(num_frames):
            if i != j:
                feature_similarity[i, j] = tf.reduce_sum(normalized_features[i] * normalized_features[j])
                feature_similarity[i, j] = (feature_similarity[i, j] + 1)/2    # Normalize to [0, 1]
            else:
                feature_similarity[i, j] = 0 # No self-similarity

    # 2. Temporal_similarity:  
    # Compute the temporal similarity based on the distance between frames
    for i in range(num_frames):
        for j in range(num_frames):
            if i != j:
                temporal_similarity[i, j] = tf.exp(- beta * tf.abs(i - j))
            else:
                temporal_similarity[i, j] = 0 # No self-similarity

    # 3. Combine the two similarity matrices with a weighted sum
    similarity_matrix = (1 - alpha) * feature_similarity + alpha * temporal_similarity

    return similarity_matrix



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