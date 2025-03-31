import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np 






def mfccs_to_graph_tensors(mfccs, adjacency_matrices):
    """
    Convert MFCC features to graph tensors using custom adjacency matrices.
    
    Args:
        mfccs: Tensor of shape [batch_size, n_frames, n_features]
        adjacency_matrices: Tensor of shape [batch_size, n_frames, n_frames] # TODO : no  batch size, since static
        
    Returns:
        List of GraphTensor objects
    """
    batch_size = tf.shape(mfccs)[0]
    graph_tensors = []
    
    for i in range(batch_size):
        # Extract single example
        features = mfccs[i]  # Shape: [n_frames, n_features]
        adjacency = adjacency_matrices[i]  # Shape: [n_frames, n_frames]
        
        # Get edges from adjacency matrix
        edges = tf.where(adjacency > 0)  # Returns indices where adjacency > 0
        
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
                    features={},
                    sizes=[tf.shape(edges)[0]],
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("frames", sources),
                        target=("frames", targets)
                    )
                )
            }
        )
        
        graph_tensors.append(graph_tensor)
    
    return graph_tensors





def main():
    pass

    
    


if __name__ == '__main__':
    main()