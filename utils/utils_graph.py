import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy as sp


#TODO: alpha, beta and threshold are best to be tuned on a validation set
def create_adjacency_matrix(mfcc, num_frames, label, mode = 'similarity', window_size = 5, window_size_cosine = 10, cosine_window_thresh = 0.3, alpha = 0.7, beta = 0.1, threshold = 0.3):
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

    adjacency_matrix = tf.zeros((num_frames, num_frames), dtype=tf.float32)
    
    if mode == 'window':
        # Create a sliding window adjacency matrix based on the window size

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

    
    elif mode == 'cosine window':
        
        # Same as above, but we add edge weights based on the normalized cosine similarity between frames

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



    elif mode == 'similarity':
        # Create a similarity adjacency matrix based on the cosine similarity between frames, with a penalty for distance
        # This should ideally cluster close frames with similar frequencies together, allowing for an identification of the phonemes/words
        similarity_matrix = similarity_function(mfcc, num_frames, alpha=alpha, beta=beta)
        adjacency_matrix = tf.where(similarity_matrix >= threshold, similarity_matrix, adjacency_matrix)

    
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    



    return mfcc, adjacency_matrix , label




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






def dilated_adjacency_matrix_with_weights(dilated_adjacency_matrix, mfccs, num_frames):

    """
    Add weights to the dilated adjacency matrix based on the cosine similarity between node features.

    """

    similarity_matrix = similarity_function(mfccs, num_frames, alpha=0.7, beta=0.1)
    dilated_adjacency_matrix = tf.where(dilated_adjacency_matrix > 0, similarity_matrix, dilated_adjacency_matrix)

    return dilated_adjacency_matrix

#TODO: There are other ways to assign weights to the edges! 




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
    


#TODO: I need to choose a reference graph for the layout if I want to have comparable graphs throughout the dataset

#Here are some examples of layouts:

def grid_node_layout(G):
    pos = {}
    for node in G.nodes():
        x = node % 10  # Creates a grid with 10 columns
        y = node // 10
        pos[node] = (x, y)
    return pos
# Disadvantage: we are able to see the connections between all the nodes (the ones aligned in the same row)

def node_layout(G):
    # pos = nx.random_layout(G, seed = 123)  # Random layout
    pos = nx.spring_layout(G, k=0.15, iterations=25, weight = 'weight', seed=42)  # force-directed algorithm (bigger weights -> closer nodes)
    # pos = nx.circular_layout(G)  # Circular layout
    # pos = nx.spectral_layout(G)  # Spectral layout
    # pos = nx.multipartite_layout(G, subset_key= 10)
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