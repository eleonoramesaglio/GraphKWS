import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy as sp



def create_adjacency_matrix(mfcc, num_frames, label, mode = 'similarity', n_dilation_layers = 0, window_size = 5, window_size_cosine = 10, cosine_window_thresh = 0.3, alpha = 0.7, beta = 0.1, threshold = 0.3):
    """
    Create a custom adjacency matrix for the graph.
    Since all our MFCCs are of the same length, we can create a static adjacency matrix.
    The adjacency matrix is used to define the connections between nodes in the graph.
    The adjacency matrix is created based on the mode specified.
    
    Args:
        mfcc: The MFCCs of the audio file.
        num_frames: Number of frames in the MFCC.
        label: The label of the audio file.
        mode: Sets the mode in which the adjacency matrix is created.
               'window', 'cosine window', 'similarity'
        n_dilation_layers: Number of dilation layers to create.
        window_size: Size of the sliding window for the 'window' mode.
        window_size_cosine: Size of the sliding window for the 'cosine window' mode.
        alpha: Trade-off parameter for the similarity function.
        beta: Scaling factor in the distance penalty for the similarity function (decides how fast the penalty increases with distance).
        cosine_window_thresh: Threshold for the cosine window mode.
        threshold: Threshold for the similarity function.
   
        
        Returns:
        mfcc: The MFCCs of the audio file.
        adjacency_matrix: A 2D numpy array representing the adjacency matrix.
        label: The label of the audio file.
    """


    adjacency_matrices = []
    adjacency_matrix = tf.zeros((num_frames, num_frames), dtype=tf.float32)


    if mode == 'window':

    # 1. MODE 'WINDOW' : Creates an adjacency matrix based on a sliding window over the frames.
    #    This means that each frame is connected to its 'window_size' neighbors.
    #    The adjacency matrix is unweighted (0 or 1) and, thanks to the fixed window size, the corresponding graph is homogenous.
    #    This is our base representation.
    #    Models that are based on this representation: 
    #    - base_gnn_model
    #    (Note: we can still use all the other models with this representation, but they are not optimized for it)


        indices = tf.range(num_frames, dtype=tf.int32)

        # Create a column & row vector of the indices
        i = tf.reshape(indices, [-1, 1])
        j = tf.reshape(indices, [1,-1])

        # Calculate the absolute distance between numbers ; creates a matrix of distances
        # in size of (num_frames, num_frames)
        distance = tf.abs(i-j)

        # Based on that distance, create the adjacency matrix (by casting to 1 or 0)
        adjacency_matrix = tf.cast(distance <= window_size, dtype=tf.float32)

        # Fix self loops by setting diagonal to 0
        adjacency_matrix = tf.linalg.set_diag(adjacency_matrix, tf.zeros(num_frames, dtype=tf.float32))


        adjacency_matrices.append(adjacency_matrix)

        dilation_rate = 2

        for i in range(n_dilation_layers):
            adjacency_matrix_dilated = create_dilated_adjacency_matrix(adjacency_matrix, dilation_rate=dilation_rate)
            # Substitute the weights of the edges with the cosine similarity values
         #   similarity_matrix = normalized_cosine_similarity(mfcc, num_frames)
         #   adjacency_matrix_dilated = tf.where(adjacency_matrix_dilated > 0, similarity_matrix, adjacency_matrix_dilated)
            # Append the dilated adjacency matrix to the list
            adjacency_matrices.append(adjacency_matrix_dilated)
            # Increase the dilation rate for the next layer
            dilation_rate += 2




    

    elif mode == 'cosine window':

    # 2. MODE 'COSINE WINDOW' : Creates a weighted adjacency matrix based on a sliding window over the frames.
    #    The weights are based on the cosine similarity between the MFCCs of the frames.
    #    Eventually, the adjacency matrix can be thresholded based on 'cosine_window_thresh', to filter out weak connections.
    #    Thus, the corresponding graph is not necessarily homogenous.
    #    This mode gives more importance to the cosine similarity between frames with respect to mode 'similarity'.
    #    We introduced this representation because cosine similarity helps in handling noise, by associating low weights to the
    #    edges connecting frames containing spoken words and frames containing just background noise.
    #    Models that are based on this representation:
    #    - ...


        indices = tf.range(num_frames, dtype=tf.int32)
        i = tf.reshape(indices, [-1, 1])
        j = tf.reshape(indices, [1,-1])
        distance = tf.abs(i-j)
        adjacency_matrix = tf.cast(distance <= window_size_cosine, dtype=tf.float32)
        adjacency_matrix = tf.linalg.set_diag(adjacency_matrix, tf.zeros(num_frames, dtype=tf.float32))

        # Compute the normalized cosine similarity between all pairs of frames
        similarity_matrix = normalized_cosine_similarity(mfcc, num_frames)

        # Substitute the weights of the edges with the similarity values
        adjacency_matrix = tf.where(adjacency_matrix > 0, similarity_matrix, adjacency_matrix)

        # Now use a threshold to remove edges with low similarity
        adjacency_matrix = tf.where(adjacency_matrix >= cosine_window_thresh, adjacency_matrix, tf.zeros_like(adjacency_matrix))

        # Create list of adjacency matrices
        adjacency_matrices.append(adjacency_matrix)

        # (If n_dilation_layers > 0)
        # Create a dilated version of the adjacency matrix
        # We start with a dilation rate of 2 and increase it by 2 for each layer
        # (2, 4, 6, 8, ...)
        dilation_rate = 2

        for i in range(n_dilation_layers):
            adjacency_matrix_dilated = create_dilated_adjacency_matrix(adjacency_matrix, dilation_rate=dilation_rate)
            # Substitute the weights of the edges with the cosine similarity values
            similarity_matrix = normalized_cosine_similarity(mfcc, num_frames)
            adjacency_matrix_dilated = tf.where(adjacency_matrix_dilated > 0, similarity_matrix, adjacency_matrix_dilated)
            # Append the dilated adjacency matrix to the list
            adjacency_matrices.append(adjacency_matrix_dilated)
            # Increase the dilation rate for the next layer
            dilation_rate += 2



    elif mode == 'similarity':

    # 3. MODE 'SIMILARITY' : Creates a weighted adjacency matrix based on a similarity function between frames.
    #    The weights are based on the cosine similarity between the MFCCs of the frames, with a penalty for distance.
    #    The adjacency matrix is then thresholded based on 'threshold', to filter out connections between frames far apart.
    #    The trade-off between the cosine similarity and the distance penalty is controlled by 'alpha'.
    #    This should ideally cluster close frames with similar frequencies together, allowing for an identification of the phonemes/words.
    #    The corresponding graph is not homogenous.
    #    Models that are based on this representation:
    #    - ...

        similarity_matrix = similarity_function(mfcc, num_frames, alpha=alpha, beta=beta)
        adjacency_matrix = tf.where(similarity_matrix >= threshold, similarity_matrix, adjacency_matrix)

        # Create list of adjacency matrices
        adjacency_matrices.append(adjacency_matrix)

        # (If n_dilation_layers > 0)
        # Create a dilated version of the adjacency matrix
        # We start with a dilation rate of 2 and increase it by 2 for each layer
        # (2, 4, 6, 8, ...)
        dilation_rate = 2

        for i in range(n_dilation_layers):
            adjacency_matrix_dilated = create_dilated_adjacency_matrix(adjacency_matrix, dilation_rate=dilation_rate)
            # Substitute the weights of the edges with the similarity values
            adjacency_matrix_dilated = tf.where(adjacency_matrix_dilated > 0, similarity_matrix, adjacency_matrix_dilated)
            # Append the dilated adjacency matrix to the list
            adjacency_matrices.append(adjacency_matrix_dilated)
            # Increase the dilation rate for the next layer
            dilation_rate += 2


    
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    

    

    return mfcc, adjacency_matrices, label



def get_reduced_representation(mfccs, wav, label, k = 2, pooling_type = 'max'):
    
    """
    Reduces MFCC representation by pooling across k consecutive frames.
    
    """
    n_frames, n_features = 98, 39 # static, known values
    
    # 1. Complete groups (k elements)
    # Calculate number of complete groups of k frames
    n_complete_groups = n_frames // k
    
    if n_complete_groups > 0:
        # Select the portion of mfccs that are part of complete groups
        complete_groups = mfccs[:n_complete_groups*k]
        # Reorganize the mfccs in a tensor with dimensions [number of groups, frames per group, features per frame]
        reshaped = tf.reshape(complete_groups, [n_complete_groups, k, n_features])
        # Pool over the groups
        if pooling_type == 'max':
            complete_reduced = tf.reduce_max(reshaped, axis=1)  # take the max
        elif pooling_type == 'mean':
            complete_reduced = tf.reduce_mean(reshaped, axis=1)   # take the mean
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}. Use 'max' or 'mean'.")
    else:
        complete_reduced = tf.zeros([0, n_features], dtype=mfccs.dtype)
    
    # 2. Incomplete group (< k elements)
    # Handle the incomplete group of frames at the end (if any)
    remainder_frames = n_frames % k
    if remainder_frames > 0:
        incomplete_group = mfccs[n_complete_groups*k:]
        # Pool over the last group
        if pooling_type == 'max':
            incomplete_reduced = tf.reduce_max(incomplete_group, axis=0, keepdims=True)
        elif pooling_type == 'mean':
            incomplete_reduced = tf.reduce_mean(incomplete_group, axis=0, keepdims=True)
        
        # 3. Combine complete_reduced and incomplete_reduced
        reduced_mfccs = tf.concat([complete_reduced, incomplete_reduced], axis=0)
    else:
        reduced_mfccs = complete_reduced
    
    # Return the reduced representation
    return reduced_mfccs, wav,label



def create_dilated_adjacency_matrix(adjacency_matrix, dilation_rate = 2):

    """
    Create a dilated adjacency matrix based on the original adjacency matrix (A) to access not directly connected nodes.
    The dilation rate indicates how many hops away the nodes are.
    Such matrix is obtained with the following formula: 
        A_dilated = {A^d - [A^(d-1) + ... + A^1 + (d-1)*I]} > 0
    where d is the dilation rate and I is the identity matrix.

    """

    # Initialize the dilated adjacency matrix as the original adjacency matrix (A)
    dilated_adjacency_matrix = adjacency_matrix
    # Store the powers for more efficient computation
    powers = [adjacency_matrix]

    # Compute the matrix A^d
    for i in range(dilation_rate - 1):
        dilated_adjacency_matrix = tf.linalg.matmul(dilated_adjacency_matrix, adjacency_matrix)
        powers.append(dilated_adjacency_matrix)
    # The element a_ij in the resulting matrix indicates the number of paths of length dilation_rate from node i to node j.
    
    # Remove all powers of A from 1 to d-1 (we remove all connections less than d hops away)
    for i in range(dilation_rate - 1):
        dilated_adjacency_matrix = tf.where(powers[i] > 0, tf.zeros_like(adjacency_matrix), dilated_adjacency_matrix)

    # Remove self-loops (diagonal elements)
    dilated_adjacency_matrix = tf.linalg.set_diag(dilated_adjacency_matrix, tf.zeros(tf.shape(adjacency_matrix)[0], dtype=tf.float32))

    # Select only the positive connections
    dilated_adjacency_matrix = tf.cast(dilated_adjacency_matrix > 0, dtype=tf.float32)

  
    
    return dilated_adjacency_matrix




def normalized_cosine_similarity(mfccs, num_frames):

    """
    Compute the normalized cosine similarity between all pairs of frames in the MFCCs.

    """

    # Normalize the feature vectors for cosine similarity
    normalized_features = tf.nn.l2_normalize(mfccs, axis=1)
    # Compute the normalized cosine similarity between all pairs of frames
    cosine_similarity = tf.matmul(normalized_features, normalized_features, transpose_b=True)
    feature_similarity = (cosine_similarity + 1) / 2
    # Remove self-similarity
    identity = tf.eye(num_frames)
    feature_similarity = feature_similarity * (1 - identity)

    return feature_similarity




def similarity_function(mfccs, num_frames, alpha, beta):

    """
    Create a similarity adjacency matrix based on the cosine similarity between frames, with a penalty for distance.

    """

    # Take as input the MFCCs of a single audio file (not batched)
    # Initialize
    feature_similarity = tf.zeros((num_frames, num_frames), dtype=tf.float32)
    temporal_similarity = tf.zeros((num_frames, num_frames), dtype=tf.float32)
    similarity_matrix = tf.zeros((num_frames, num_frames), dtype=tf.float32)

    # 1. Feature_similarity:
    # Normalize the feature vectors for cosine similarity
    normalized_features = tf.nn.l2_normalize(mfccs, axis=1)
    # Compute the normalized cosine similarity between all pairs of frames
    cosine_similarity = tf.matmul(normalized_features, normalized_features, transpose_b=True)
    feature_similarity = (cosine_similarity + 1) / 2
    # Remove self-similarity
    identity = tf.eye(num_frames)
    feature_similarity = feature_similarity * (1 - identity)

    # 2. Temporal_similarity:  
    # Compute the temporal similarity based on the distance between frames
    indices = tf.range(num_frames, dtype=tf.float32)
    i = tf.reshape(indices, [-1, 1])
    j = tf.reshape(indices, [1, -1])
    distance = tf.abs(i - j)
    temporal_similarity = tf.exp(- beta * distance)
    # Remove self-similarity
    temporal_similarity = temporal_similarity * (1 - identity)

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



def convert_tensor_to_networkx(graph_tensor):
    """
    Convert a tfgnn.GraphTensor to a NetworkX graph for visualization.
    
    Args:
        graph_tensor: TensorFlow graph tensor

    Returns:
        G: NetworkX graph
    """
    # Create an empty NetworkX graph
    G = nx.Graph()

    # Extract node sets
    node_sets = graph_tensor.node_sets
    # Note 1: in our case we only have one set, but we can also have multiple node sets, and we use a different node_set_name to distinguish them
    # Note 2: node_set_name = 'frames'

    # Add nodes from each node set
    for node_set_name, node_set in node_sets.items():
        # Get the number of nodes in this node set
        num_nodes = tf.reduce_sum(node_set.sizes).numpy()
        
        # Add nodes
        for i in range(num_nodes):
            # Add node to graph
            G.add_node(i)   # (if we have different node sets, we add node_set_name as attribute)
    
    # Extract edge sets
    edge_sets = graph_tensor.edge_sets
    # Note 1: same as above, we can have multiple edge sets
    # Note 2: edge_set_name = 'connections'
    # Note 3: we also have only one type of feature, feature_name = 'weights'
    
    # Add edges from each edge set
    for edge_set_name, edge_set in edge_sets.items():
        # Get source and target node indices
        source_indices = edge_set.adjacency.source.numpy()
        target_indices = edge_set.adjacency.target.numpy()

        # Get edge weights
        edge_features = {}
        for feature_name, feature_value in edge_set.features.items():
            edge_features[feature_name] = feature_value.numpy()
        
        # Add edges and edge weights
        for i in range(len(source_indices)):
            source_idx = int(source_indices[i])
            target_idx = int(target_indices[i])

            edge_attrs = {feature_name: feature_value[i] for feature_name, feature_value in edge_features.items()}
             # (if we have different edge sets, we add edge_set_name as attribute)

            # Add edge to graph
            G.add_edge(source_idx, target_idx, **edge_attrs)
    
    return G
    



def node_layout(G):
    pos = nx.spring_layout(G, k=0.15, iterations=25, weight = 'weight', seed=42)  # force-directed algorithm (bigger weights -> closer nodes)
    # pos = nx.circular_layout(G)  # Circular layout
    # pos = nx.spectral_layout(G)  # Spectral layout
    # pos = nx.random_layout(G, seed = 123)  # Random layout
    return pos


def visualize_graph(G, pos, title):
    """
    
    Parameters:
    - G: NetworkX graph
    - pos: optional pre-computed positions """


    fig = plt.figure(figsize=(12, 8))
    
    # If no position provided, use a random layout
    if pos is None:
        pos = nx.random_layout(G, seed = 123)

    # Define the node colors
    node_colors = []
    node_color_map = ['lightblue', 'lightgreen', 'lavender', 'lightpink', 'lightyellow']

    # Assign colors: first 20 nodes get the first color, next 20 get the second color, etc.
    for i, node in enumerate(sorted(G.nodes())):
        color_index = color_index = (i // 20) % len(node_color_map)
        node_colors.append(node_color_map[color_index])

    # Draw the graph
    nx.draw(G, pos, node_color= node_colors, edge_color='grey', font_size=10, font_weight='bold')

    # Add custom labels (frames from 1 to 98)
    custom_labels = {node: str(int(node) + 1) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=10)

    # Add title
    fig.suptitle(title, fontsize=16, y=0.95)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def visualize_graph_with_heatmap(G, pos, title):

    fig = plt.figure(figsize=(12, 8))
    
    # If no position provided, use a random layout
    if pos is None:
        pos = nx.random_layout(G, seed = 123)

    # Define the node colors
    node_colors = []
    node_color_map = ['lightblue', 'lightgreen', 'lavender', 'lightpink', 'lightyellow']

    # Assign colors: first 20 nodes get the first color, next 20 get the second color, etc.
    for i, node in enumerate(sorted(G.nodes())):
        color_index = color_index = (i // 20) % len(node_color_map)
        node_colors.append(node_color_map[color_index])

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)

    # Add custom labels (frames from 1 to 98)
    custom_labels = {node: str(int(node) + 1) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=10)
    
    # Get edge weights
    edges = G.edges()
    weights = [G[u][v]['weights'] for u, v in edges]
    
    # Create a colormap - darker=higher weight
    cmap = plt.cm.Blues
    
    # Draw edges with colors based on weights
    edges = nx.draw_networkx_edges(
        G, pos,
        edge_color=weights,
        width=2,
        edge_cmap=cmap,
        edge_vmin=0,
        edge_vmax=1,
        alpha=0.7
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax = ax, shrink=0.6, pad=0.05)
    cbar.set_label('Edge Weight')

    # Add title
    fig.suptitle(title, fontsize=16, y=0.95)
    plt.axis('off')
    plt.tight_layout()
    plt.show()