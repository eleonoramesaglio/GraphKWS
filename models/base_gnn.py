import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np 



def mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrix, label):


    # Extract single example
    features = mfcc # Shape: [n_frames, n_features]
    adjacency = adjacency_matrix  # Shape: [n_frames, n_frames]
    
    # Get edges from adjacency matrix
    edges = tf.where(adjacency > 0)  # Returns indices where adjacency > 0

    # Get corresponding weights of edges from adjacency matrix
    # e.g. edges has saved [0,3] --> goes into adjacency[0,3] and gets the weight
    weights = tf.gather_nd(adjacency, edges)  
    
    # The edges tensor has shape [num_edges, 2] where each row is [source, target]
    sources = edges[:, 0]
    targets = edges[:, 1]
    
    # Create GraphTensor
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "frames": tfgnn.NodeSet.from_fields(
                    
                
                features={"features": features},
                sizes=[tf.shape(features)[0]]
            )
        },
        edge_sets={
            "connections": tfgnn.EdgeSet.from_fields(
                features={"weights" : weights}, # possibly here just adjacency ?????
                # but I think okay like this, since the adjacency below defines which edges
                # exist and then by normal indexing, it takes the correct weights
                # per edge
                sizes=[tf.shape(edges)[0]],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("frames", sources),
                    target=("frames", targets)
                )
            )
        }
    )
        
 
    
    return graph_tensor, label





def mfccs_to_graph_tensors(mfccs, adjacency_matrices):
    """
    Convert MFCC features to graph tensors using custom adjacency matrices.
    
    Args:
        mfccs: Tensor of shape [batch_size, n_frames, n_features]
        adjacency_matrices: Tensor of shape [batch_size, n_frames, n_frames]
        
    Returns:
        List of GraphTensor objects
    """
    # Extract single example
    features = mfccs  # Shape: [n_frames, n_features]
    adjacency = adjacency_matrices  # Shape: [n_frames, n_frames]
    
    # Get edges from adjacency matrix
    edges = tf.where(adjacency > 0)  # Returns indices where adjacency > 0

    # Get corresponding weights of edges from adjacency matrix
    # e.g. edges has saved [0,3] --> goes into adjacency[0,3] and gets the weight
    weights = tf.gather_nd(adjacency, edges)  #### possibly don't use that ????
    
    # The edges tensor has shape [num_edges, 2] where each row is [source, target]
    sources = edges[:, 0]
    targets = edges[:, 1]
    
    # Create GraphTensor
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "frames": tfgnn.NodeSet.from_fields(
                
                
                features={"features": features},
                sizes=[tf.shape(features)[0]]
            )
        },
        edge_sets={
            "connections": tfgnn.EdgeSet.from_fields(
                features={"weights" : weights}, # possibly here just adjacency ?????
                # but I think okay like this, since the adjacency below defines which edges
                # exist and then by normal indexing, it takes the correct weights
                # per edge
                sizes=[tf.shape(edges)[0]],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("frames", sources),
                    target=("frames", targets)
                )
            )
        }
    )
    

    
    return graph_tensor




def create_base_gnn_model(
        input_dim,
        hidden_dim = [64,128],
        num_classes = 30,
        dropout_rate = 0.5,
        num_message_passing_layers = 3
):
    
    """
    Create a base GNN model for audio classification.
    Args:
        input_dim: Dimension of the input features ; in our case, the feature dimension of each frame, i.e. the MFCCS
        hidden_dim: List of hidden dimensions for each layer.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate for regularization.
        num_message_passing_layers: Number of message passing layers in the GNN.
    Returns:
        model: A Keras model instance.
        
    """


    # First, we define the two inputs : MFCC & adjacency matrix 

    mfcc_input = tf.keras.layers.Input(shape=(None, input_dim), name='mfcc')
    adjacency_input = tf.keras.layers.Input(shape=(None, None), name='adjacency_matrix')

    # Next, we need to convert the MFCCs and adjacency matrix to Graph Tensors 
    # which will be the actual input into the model
    # We use a lambda layer, because else we would need to convert the MFCCs and adjacency matrix to graph tensors
    # before we define the model, which is not efficient
    # The lambda layer will apply the function to each batch of MFCCs and adjacency matrices
    graph_tensors = tf.keras.layers.Lambda(lambda x: mfccs_to_graph_tensors(x[0], x[1]))([mfcc_input, adjacency_input])


    # Now we need to define the actual graph input 
    graph_input = tf.keras.layers.Input(type_spec = tfgnn.GraphTensorSpec.from_piece_specs(
        # define the node set
        node_sets = {
            'frames': tfgnn.NodeSet.from_fields(
                # define the features of the node set (i.e. the MFCCs)
                features = {'features': tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32)},
            )
        },
        # define the edge set
        edge_sets= {
            'connections': tfgnn.EdgeSet.from_fields(
                # define the adjacency matrix
                adjacency = tfgnn.Adjacency.from_indices(
                    source=('frames', tf.TensorSpec(shape=(None,), dtype=tf.int32)),
                    target=('frames', tf.TensorSpec(shape=(None,), dtype=tf.int32))
                ),
                # define the features of the edge set
                features = {}
            )
        }
    ))

    






def main():
    pass

    
    


if __name__ == '__main__':
    main()