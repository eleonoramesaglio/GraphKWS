import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np 
from tensorflow_gnn.models.gcn import gcn_conv
from tensorflow_gnn.models.gat_v2.layers import GATv2Conv
from utils import utils_metrics
#tf.config.run_functions_eagerly(True) 
#tf.data.experimental.enable_debug_mode()


"""
def mfccs_to_graph_tensors_for_dataset_OLD(mfcc, adjacency_matrices, label):

    # Print shapes to debug
  #  tf.print("MFCC shape:", tf.shape(mfcc))
  #  tf.print("Adjacency shape:", tf.shape(adjacency_matrix))
  #  tf.print("Label shape:", tf.shape(label))


    adjacency_matrix = adjacency_matrices[0]

   
    adjacency_matrices = adjacency_matrices[1:]


    # Ensure current shape of MFCC (98 frames, 39 MFCCs)
    mfcc_static = tf.reshape(mfcc, [98, 39]) 

    
    # Get edges from adjacency matrix
    edges = tf.where(adjacency_matrix > 0)  # Returns indices where adjacency > 0

    # Get corresponding weights of edges from adjacency matrix
    # e.g. edges has saved [0,3] --> goes into adjacency[0,3] and gets the weight
    weights = tf.gather_nd(adjacency_matrix, edges)  
    
    # The edges tensor has shape [num_edges, 2] where each row is [source, target]
    sources = edges[:, 0]
    targets = edges[:, 1]


    # Create GraphTensor
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "frames": tfgnn.NodeSet.from_fields(
                    
                
                features={"features": mfcc_static},  
                sizes=[tf.shape(mfcc_static)[0]]
            )
        },
        edge_sets={
            "connections": tfgnn.EdgeSet.from_fields(
                features={"weights" : weights},
                sizes=[tf.shape(edges)[0]],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("frames", sources),
                    target=("frames", targets)
                )
            )
        }
    )
        

    
    return graph_tensor, label
"""

"""
def mfccs_to_graph_tensors_multi_nodes_for_dataset(mfcc, adjacency_matrices, label):

    '''
    
    Two node sets : One normal (98 Frames) , other one 10 nodes (or can be changed
    dynamically in size) for clustering ; else everything the same
    
    '''
    # Ensure current shape of MFCC (98 frames, 39 MFCCs)
    mfcc_static = tf.reshape(mfcc, [98, 39])

    cluster_features_static = tf.ones([10,1], dtype = tf.float32)

    # Create 
    
    # Create the node set that will be shared by all edge sets
    node_sets = {
        "frames": tfgnn.NodeSet.from_fields(
            features={"features": mfcc_static},  
            sizes=[tf.shape(mfcc_static)[0]]
        ),
        "clusters" : tfgnn.NodeSet.from_fields(
            features = {"features" : cluster_features_static},
            sizes = [tf.shape(cluster_features_static)[0]]
        )
    }
    
    # Create an edge set for each adjacency matrix
    edge_sets = {}
    

    # Create the directed 98x10 (i.e. just from the 98 nodes to the 10)

    adj_matrix_f_to_c = tf.ones(shape = (98,10))

    edges_f_to_c = tf.where(adj_matrix_f_to_c > 0)

    weights_f_to_c = tf.gather_nd(adj_matrix_f_to_c, edges_f_to_c)

    sources_f_to_c = edges_f_to_c[:, 0]
    targets_f_to_c = edges_f_to_c[:, 1]



    edge_sets["frames_to_clusters"] = tfgnn.EdgeSet.from_fields(
        features = {"weights" : weights_f_to_c},
        sizes = [tf.shape(edges_f_to_c)[0]],
        adjacency = tfgnn.Adjacency.from_indices(
            source =("frames", sources_f_to_c),
            target = ("clusters", targets_f_to_c)
        )

    )

    # Unstack the matrices so we can iterate over them
    unstacked_matrices = tf.unstack(adjacency_matrices, axis=0)

    for i, adjacency_matrix in enumerate(unstacked_matrices):
        # Get edges from this adjacency matrix
        edges = tf.where(adjacency_matrix > 0)
        
        # Get corresponding weights
        weights = tf.gather_nd(adjacency_matrix, edges)

      #  weights = tf.reshape(weights, [-1, 1])
        
        # Extract source and target indices
        sources = edges[:, 0]
        targets = edges[:, 1]
        
        # Create edge set with unique names
        edge_set_name = f"connections_{i}"
        
        edge_sets[edge_set_name] = tfgnn.EdgeSet.from_fields(
            features={"weights" : weights}, 
            sizes=[tf.shape(edges)[0]],
            adjacency=tfgnn.Adjacency.from_indices(
                source=("frames", sources),
                target=("frames", targets)
            )
        )
    
    # Create the graph tensor with all node sets and edge sets
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets,
        edge_sets=edge_sets
    )
    
    return graph_tensor, label
"""


def mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrices, label, reduced_node_bool, reduced_node_k):
    """
    Convert MFCC (or GFCCs) features and adjacency matrices to a graph tensor where
    each adjacency matrix becomes a separate edge set in the graph.
    
    Args:
        mfcc: MFCC (or GFCC) features (features of shape [batch_size, n_frames, n_features])
        adjacency_matrices: List of adjacency matrices, each will become an edge set
        label: Class label
    
    Returns:
        A tuple (graph_tensor, label) where graph_tensor contains multiple edge sets
    """
  
    if reduced_node_bool:
        if ((98 // reduced_node_k) == (98/ reduced_node_k)):
          mfcc_static = tf.reshape(mfcc, [98 // reduced_node_k, 39])
        else:
          # this case when we have some overhang of a group 
          mfcc_static = tf.reshape(mfcc, [((98 // reduced_node_k) + 1), 39])
    else:
        mfcc_static = tf.reshape(mfcc, [98, 39])
    
    # Create the node set that will be shared by all edge sets
    node_sets = {
        "frames": tfgnn.NodeSet.from_fields(
            features={"features": mfcc_static},  
            sizes=[tf.shape(mfcc_static)[0]]
        )
    }
    
    # Create an edge set for each adjacency matrix
    edge_sets = {}
    
    # Unstack the matrices so we can iterate over them
    unstacked_matrices = tf.unstack(adjacency_matrices, axis=0)

    for i, adjacency_matrix in enumerate(unstacked_matrices):
        # Get edges from this adjacency matrix
        edges = tf.where(adjacency_matrix > 0)
        
        # Get corresponding weights
        weights = tf.gather_nd(adjacency_matrix, edges)

      #  weights = tf.reshape(weights, [-1, 1])
        
        # Extract source and target indices
        sources = edges[:, 0]
        targets = edges[:, 1]
        
        # Create edge set with unique names
        edge_set_name = f"connections_{i}"
        
        edge_sets[edge_set_name] = tfgnn.EdgeSet.from_fields(
            features={"weights" : weights}, 
            sizes=[tf.shape(edges)[0]],
            adjacency=tfgnn.Adjacency.from_indices(
                source=("frames", sources),
                target=("frames", targets)
            )
        )
    
    # Create the graph tensor with all node sets and edge sets
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets=node_sets,
        edge_sets=edge_sets
    )
    
    return graph_tensor, label


# Function is used if we work with single samples for testing, not with the dataset ; deprecated 
def mfccs_to_graph_tensors(mfccs, adjacency_matrices):
    """
    Convert MFCC/GFCC features to graph tensors using custom adjacency matrices.
    
    Args:
        mfccs: MFCCs/GFCCs (features of shape [batch_size, n_frames, n_features])
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
    

    
    return graph_tensor



def base_gnn_model(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',
        use_residual_next_state = False,


        ):
    
    """
    Our very Base GNN model. Uses the simple window adajcency matrix (i.e. unweighted approach), simple message 
    aggregation using SimpleConv() and NextStateFromConcat() to compute the next state of the nodes.

    Note that we wrote a lot of comments here to understand the code better (i.e. followed tutorials provided by TensorFlow)
    """


    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)


    ### IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape []
    ### in which the graphs of the input batch have been merged to components of
    ### one contiguously indexed graph. There are no edges between components,
    ### so no information flows between them.
    if is_batched:
        batch_size = graph.shape[0]
        #merge all graphs of the batch into one, contiguously indexed graph.
        #  The resulting GraphTensor has shape [] (i.e., is scalar) and its features h
        # ave the shape [total_num_items, *feature_shape] where total_num_items is the sum 
        # of the previous num_items per batch element. At that stage, the GraphTensor is ready
        #  for use with the TF-GNN model-building code, but it is no longer easy to split it up.
        # https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/graph_tensor.md
        # this means our nodes [32,98,39] are now [32*98,39] = [3136,39]
        graph = graph.merge_batch_to_components()





    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

                # TODO : try to do base mfcc + its energy, delta + energy, delta-delta + energy
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
        

            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
        pass 


    def set_initial_context_state():
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

        pass
        

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state, name = 'init_states')(graph)
    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization())
        return result 
    

    # Without edge weights
    # If we have a unweighted adjacency matrix (window case), the function convolution_with_weights will return the same result as the normal convolution
    # If instead we have a weighted adjacency matrix, convolution_with_weights implements a sort of "attention mechanism", giving more importance
    # to the edges with higher weights.

    #This layer can compute a convolution over an edge set by applying the
    #  passed-in message_fn (dense(message_dim) here) for all edges on the 
    # concatenated inputs from some or all of: the edge itself, the sender node, and the receiver node, followed by pooling to the receiver node.


    # message_fn : layer that computes the individual messages after they went through the "combine_type" aggregation (i.e. here rn from our target & source node the embeddings?)
    # combine_type : defines how to combine the messages before passing them through the message layer (i.e concat, sum (element-wise) etc.)
    # --> this combines right now the node features (of target & source node of the corresponding edge //
    # NO : just the target or source node, depending opn what the receiver tag is (so doesn't use the information of the node
    # that it is sending to)) and also edge features etc. if they are any (just combines all of them)
    # reduce_type : Specifices how to combine the messages (of ALL the nodes) after passing them through the message layer (max/min/mean...)
    # receiver_tag : defines the receiver of those messages (i.e. here in our implementation which node receives them)
    # receiver_tag  : could also be context node here and then we pool information into the context node!!! read the documentation on simpleconv!


    def convolution(message_dim, receiver_tag):


        # Set receiver feature to None such that we don't concatenate the receiver node's features (we do that in nextstatefromconcat)
        return tfgnn.keras.layers.SimpleConv(dense(message_dim), "sum", receiver_tag = receiver_tag, receiver_feature= None)
    
    # Function: AGGREGATION
    # The convolution function is used to AGGREGATE messages from the neighbors of a node to update its state.
    # There are two functions deciding the type of aggregation: reduce_type and combine_type.
    # The reduce_type specifies how to combine the messages from the neighbors (e.g., sum, mean, max).
    # The combine_type is instead used when there are multiple messages (deriving from multiple features) from the same neighbor.
    # It decides how to aggregate these messages (usually they are concatenated) before passing them to the receiver node.
    # The receiver_tag specifies the node that will receive the aggregated messages.

    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    


    # Function: COMPUTE NEXT STATE
    # The next_state function is used to update the state of a node after aggregating messages from its neighbors.
    # 1. Concatenates the node's current state with the aggregated messages
    # 2. Processes them through a dense layer with regularization & normalization to produce the new state.
    # 3. Returns the new state for the node, with dimensions specified by next_state_dim.
    # This function is then used in the NodeSetUpdate layer to update the node states.

   
    

    # The GNN "core" of the model 
    # Convolutions let data flow towards the specified endpoint of edges
    # Note that the order here defines how the updates happen (so first e.g. NodeSetUpdate, then 
    # EdgeSetUpdate etc. (? is that true or in parallel ?))
    # NodeSetUpdate : receives the input graph and returns a new hidden state for the node set it gets applied
    # to. The new hidden state is computed with the given next-state layer from the node set's prior state and the
    # aggregated results from each incoming edge set

    # For example, each round of this model computes a new state for the node set "frames" by applying 
    # dense(next_state_dim) (i.e. the next_state function) to the concatenation of (since we do 
    # NextStateFromConcat) the result of convolution(message_dim)(graph, edge_set_name= "connections")
    # (i.e. here we dont even need to concat because we just have one set of edges I believe)

    # A convolution on an edge set computes a value for each edge (a "message") as a trainable function of the node states
    # at both endpoints (of the edge) and then aggregates the results at the receiver nodes by forming the sum
    # (or mean or max) (i.e. that is the aggregation method in convolution) over all incoming edges

    # For example, the convolution on edge set "connections" concatenates the node state of each edge's incident
    # "node1" & "node2" (??) node, applies dense(message_dim) (so I guess since we use dense layer of size 64
    # for the nodes, when we concatenate two we have 64+64 = 128 and therefore message_dim needs to be 128 ?)
    # and sums (or avgs, max,...) the results over the edges incident to each SOURCE node (this means at the SOURCE
    # node, all incoming messages are summed (or avgs,max,...) together!)
    # I think for us we could do source or target, since we have anyway undirected edges (i.e. if node a & b
    # are connected, they will both appear once as source and once as target)
    

    ### From : https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_indepth.ipynb#scrollTo=jd02cyRB5DP1
    #Notice that the conventional names *source* and *target* for the endpoints of a directed edge
    #  do **not** prescribe the direction of information flow: each "written" edge logically goes from a 
    # paper to its author (so the "author" node is its `TARGET`), yet this model lets the data flow towards 
    # the paper (and the "paper" node is its `SOURCE`). In fact, sampled subgraphs have edges directed away
    #  from the root node, so data flow towards the root often goes from `TARGET` to `SOURCE`.



    #The code below creates fresh Convolution and NextState layer objects for each edge set and node set, 
    # resp., and for each round of updates. This means they all have separate trainable weights. If
    #  desired, weight sharing is possible in the standard Keras way by sharing convolution and 
    # next-state layer objects, provided the input sizes match.

    #For more information on defining your own GNN models (including those with edge and context states), 
    # please refer to the [TF-GNN Modeling Guide](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/gnn_modeling.md).


    # n_message_passing_layers defines from how far in neighbour terms we are getting information (i.e. when 4, this means any node in the graph
    # accumulates information from 4 neighbours away (. -- . -- . -- . -- .) : Node 1 has some info of Node 5 embedded in itself.)

    # Note that initially, the nodes are (3136,39) and after the first message passing layer, they are (3136,128) !

    if not dilation:
        # Like this, in the modulo calculation, we only use connections_0 all the time, i.e. we do not use dilation
        n_dilation_layers = 1


    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : convolution(message_dim, tfgnn.SOURCE)},
                next_state(next_state_dim, use_layer_normalization)
                )
            }
        )(graph)



    # Take all the 98 learnt node features , aggregate them using sum
    # which is then representing the context vector (i.e. the "graph node")
    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, context_mode, node_set_name = "frames")(graph)  
    logits = tf.keras.layers.Dense(num_classes)(pooled_features)


    
    model = tf.keras.Model(input_graph, logits)


    return model 


def base_gnn_model_learning_edge_weights(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = 64,
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        use_residual_next_state = False,
        n_dilation_layers = 2,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',


        ):
    
    """

    The base GNN model, not utilizing the pre-calculated weights of the adjacency matrix,
    but learning the weights during training.

    """

    

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)



    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()


    def map_edge_features(edge_set, edge_set_name):
        if edge_set_name == "connections_0":
            return {"weights": tf.expand_dims(edge_set["weights"], axis=-1)}
        return edge_set.features

    graph = tfgnn.keras.layers.MapFeatures(
        edge_sets_fn=map_edge_features
    )(graph)


    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

            
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
            
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph.
        Right now we just learn for the non-dilated adjacency matrix (i.e. connections_0)
        """
        if edge_set_name == "connections_0":

            return tf.keras.layers.Dense(initial_edges_weights_layer_dims, activation="relu")(edge_set['weights'])

        else:
            # Handle any other edge types
            raise ValueError(f"Unknown node set: {edge_set_name}")


    def set_initial_context_state():
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

        pass
        

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        edge_sets_fn = set_initial_edge_state,
          name = 'init_states')(graph)
    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization())
        return result 
    


    def convolution(message_dim, receiver_tag):
        return tfgnn.keras.layers.SimpleConv(dense(message_dim), "sum", receiver_tag = receiver_tag,
                                             sender_edge_feature= tfgnn.HIDDEN_STATE, receiver_feature= None) # Sender edge feature needed here for edge learning
    

    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    


    def next_state_concat(next_state_dim, use_layer_normalization):

        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))


    if not dilation:
        # Like this, in the modulo calculation, we only use connections_0 all the time, i.e. we do not use dilation
        n_dilation_layers = 1


    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        # https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/tfgnn/keras/layers/NodeSetUpdate.md
        graph = tfgnn.keras.layers.GraphUpdate(

            #https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/tfgnn/keras/layers/EdgeSetUpdate.md
            #selects input features from the edge and its incident nodes, then passes them through a next-state layer
            edge_sets = {
                "connections_0" : tfgnn.keras.layers.EdgeSetUpdate(
                    next_state = next_state_concat(next_state_dim, use_layer_normalization),
                    edge_input_feature = tfgnn.HIDDEN_STATE
                  
                )
            },

            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : convolution(message_dim, tfgnn.SOURCE)},
                next_state(next_state_dim, use_layer_normalization)
                )
            },
        )(graph)




    # Take all the 98 learnt node features , aggregate them using sum
    # which is then representing the context vector (i.e. the "graph node")
    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, context_mode, node_set_name = "frames")(graph)  
    logits = tf.keras.layers.Dense(num_classes)(pooled_features)


    
    model = tf.keras.Model(input_graph, logits)


    return model 


def GAT_GCN_model_v2(
        
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_residual_next_state = False,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        per_head_channels = 128,
        num_heads = 2,
        initial_state_mfcc_mode = 'normal'


        ):
    

    """ 
    GAT for context node, GCN for node features 
        GAT for context node after the message passing layers
        
    """


    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)



    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()





    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

                
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
        
        pass 


    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

        # Option 1 : initialize the context node with a zero vector
        return tfgnn.keras.layers.MakeEmptyFeature()(context)
        

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        context_fn= set_initial_context_state, name = 'init_states')(graph)
    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization())
        return result 
    

    def gat_convolution(num_heads, receiver_tag):
        # Here we now use a GAT layer

        regularizer = tf.keras.regularizers.l2(l2_reg_factor)


        return  GATv2Conv(
            num_heads = num_heads,
            per_head_channels = per_head_channels, # dimension of vector of output of each head
            heads_merge_type = 'concat', # how to merge the heads
            receiver_tag = receiver_tag, # also possible nodes/edges ; see documentation of function !
            receiver_feature = tfgnn.HIDDEN_STATE,
            sender_node_feature = tfgnn.HIDDEN_STATE,
            sender_edge_feature= None,
            kernel_regularizer= regularizer,

        )
    

    def gcn_convolution(message_dim, receiver_tag):

        regularizer = tf.keras.regularizers.l2(l2_reg_factor)


        return  gcn_conv.GCNConv(
            units = message_dim,
            receiver_tag= receiver_tag,
            activation = "relu",
            use_bias = True,
            kernel_regularizer = regularizer,
            add_self_loops = False,
            edge_weight_feature_name= 'weights',
            degree_normalization= 'in'
        )


    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        

    def next_state_concat(next_state_dim, use_layer_normalization):
        # Needed for the context update
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))

    



    if not dilation:
        # Like this, in the modulo calculation, we only use connections_0 all the time, i.e. we do not use dilation
        n_dilation_layers = 1 


    
    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : gcn_convolution(message_dim, tfgnn.TARGET)},
                next_state(next_state_dim, use_layer_normalization)
                )
            },


            
        )(graph)


    # Finally, update context node
    graph = tfgnn.keras.layers.GraphUpdate(
            context = tfgnn.keras.layers.ContextUpdate(
                {
                    "frames" : gat_convolution(num_heads= num_heads, receiver_tag = tfgnn.CONTEXT)
                },
                next_state_concat(next_state_dim, use_layer_normalization)
    )
    )(graph)

    # Get the current context state (has shape (batch_size, 128) , where 128 is the message_passing_dimension)
    # This represents the master node, which is updated in each message passing layer !
    context_state = graph.context.features['hidden_state']

    logits = tf.keras.layers.Dense(num_classes)(context_state)

    model = tf.keras.Model(input_graph, logits)

    return model 


def GAT_GCN_model(
        
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        use_residual_next_state = False,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        per_head_channels = 128,
        num_heads = 2,
        initial_state_mfcc_mode = 'normal',


        ):
    

    """ GAT for context node, GCN for node features ; GAT in each message passing layer for the context node """


    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)



    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()





    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

              
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
       
        pass 


    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

       
        return tfgnn.keras.layers.MakeEmptyFeature()(context)
        

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        context_fn= set_initial_context_state, name = 'init_states')(graph)
    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization())
        return result 
    

    def gat_convolution(num_heads, receiver_tag):
        # Here we now use a GAT layer

        regularizer = tf.keras.regularizers.l2(l2_reg_factor)


        return  GATv2Conv(
            num_heads = num_heads,
            per_head_channels = per_head_channels, # dimension of vector of output of each head
            heads_merge_type = 'concat', # how to merge the heads
            receiver_tag = receiver_tag, # also possible nodes/edges
            receiver_feature = tfgnn.HIDDEN_STATE,
            sender_node_feature = tfgnn.HIDDEN_STATE,
            sender_edge_feature= None,
            kernel_regularizer= regularizer,

        )
    

    def gcn_convolution(message_dim, receiver_tag):

        regularizer = tf.keras.regularizers.l2(l2_reg_factor)


        return  gcn_conv.GCNConv(
            units = message_dim,
            receiver_tag= receiver_tag,
            activation = "relu",
            use_bias = True,
            kernel_regularizer = regularizer,
            add_self_loops = False,
            edge_weight_feature_name= 'weights',
            degree_normalization= 'in'
        )


    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        

    def next_state_concat(next_state_dim, use_layer_normalization):
        # Needed for the context update
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    



    if not dilation:
        # Like this, in the modulo calculation, we only use connections_0 all the time, i.e. we do not use dilation
        n_dilation_layers = 1 


    
    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : gcn_convolution(message_dim, tfgnn.TARGET)},
                next_state(next_state_dim, use_layer_normalization)
                )
            },

            context = tfgnn.keras.layers.ContextUpdate(
                {
                    "frames" : gat_convolution(num_heads= num_heads, receiver_tag = tfgnn.CONTEXT)
                },
                next_state_concat(next_state_dim, use_layer_normalization)
            
        ))(graph)



    # Get the current context state (has shape (batch_size, 128) , where 128 is the message_passing_dimension)
    # This represents the master node, which is updated in each message passing layer !
    context_state = graph.context.features['hidden_state']

    logits = tf.keras.layers.Dense(num_classes)(context_state)

    model = tf.keras.Model(input_graph, logits)

    return model 


# NOTE : Not included in the paper
def base_GATv2_model(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        use_residual_next_state = False,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        per_head_channels = 128,
        num_heads = 2,
        initial_state_mfcc_mode = 'normal',




        ):
    
    """
        using GATv2. 

        https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/models/gat_v2/layers.py

        Notice how this implements its own attention mechanism on edge weights. This means,
        in each message passing layer, the node receives messages from multiple different
        nodes , but unlike in our current implementation, these weights are not static (
        so not just adjacency 1 or 0 or weighted temporally & in similarity) but are learnt.
        Therefore, we use here our normal, unweighted adjacency matrix ! (set in main
        mode to "window" 

        In this basic implementation, we use the attention mechanism to gather information from our
        nodes hidden states into the context 
        node, i.e. we initialize in the beginning a context node.
        Here, much more is possible ; look into the documentation !

        using also weighted adjacency matrix for in-between nodes


    """


    

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)



    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()



    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

             
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")



    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
        
        pass 


    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

        
        return tfgnn.keras.layers.MakeEmptyFeature()(context)
        

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        context_fn= set_initial_context_state, name = 'init_states')(graph)
    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization())
        return result 
    

    def gat_convolution(num_heads, receiver_tag):
        # Here we now use a GAT layer

        regularizer = tf.keras.regularizers.l2(l2_reg_factor)


        return  GATv2Conv(
            num_heads = num_heads,
            per_head_channels = per_head_channels, # dimension of vector of output of each head
            heads_merge_type = 'concat', # how to merge the heads
            receiver_tag = receiver_tag, # also possible nodes/edges ; see documentation of function !
            receiver_feature = tfgnn.HIDDEN_STATE,
            sender_node_feature = tfgnn.HIDDEN_STATE,
            sender_edge_feature= None,
            kernel_regularizer= regularizer,


        )
    

    class WeightedSumConvolution(tf.keras.layers.Layer):

        def __init__(self, message_dim, receiver_tag):
            super().__init__()
            self.message_dim = message_dim
            self.receiver_tag = receiver_tag
            self.sender_tag = tfgnn.SOURCE if receiver_tag == tfgnn.TARGET else tfgnn.TARGET
            self.dense = dense(units = message_dim, use_layer_normalization = use_layer_normalization)
        
        def call(self, graph, edge_set_name):
            # Get node states
            messages = tfgnn.broadcast_node_to_edges(
                graph,
                edge_set_name,
                self.sender_tag,
                feature_name="hidden_state") # Take the hidden state of the node
            
            # Get edge weights
            weights = graph.edge_sets[edge_set_name].features['weights']
            
            # Apply weights to messages
            weighted_messages = tf.expand_dims(weights, -1) * messages
            
            # Pool messages to target nodes
            pooled_messages = tfgnn.pool_edges_to_node(
                graph,
                edge_set_name,
                self.receiver_tag,
                reduce_type='sum',
                feature_value=weighted_messages)
            
            # Transform pooled messages
            return self.dense(pooled_messages)


    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        

    def next_state_concat(next_state_dim, use_layer_normalization):
        # Needed for the context update
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    

    if not dilation:
        n_dilation_layers = 1
    
    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : WeightedSumConvolution(message_dim, tfgnn.TARGET)},
                next_state(next_state_dim, use_layer_normalization)
                )
            },

            context = tfgnn.keras.layers.ContextUpdate(
                {
                    "frames" : gat_convolution(num_heads= num_heads, receiver_tag = tfgnn.CONTEXT)
                },
                next_state_concat(next_state_dim, use_layer_normalization)
            
        ))(graph)



    # Get the current context state (has shape (batch_size, 128) , where 128 is the message_passing_dimension)
    # This represents the master node, which is updated in each message passing layer !
    context_state = graph.context.features['hidden_state']



    logits = tf.keras.layers.Dense(num_classes)(context_state)


    
    model = tf.keras.Model(input_graph, logits)

    return model 


def base_gnn_model_using_gcn(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        mode = 'layer',
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        use_residual_next_state = False,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',



        ):
    
    """

    In this approach, instead of using SimpleConv() for the message passing, we use GCN layers.

    """


    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()



    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

                
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
        # TODO : can be implemented if we want 
        pass 


    def set_initial_context_state():
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """
        # TODO : can be implemented if we want
        pass
        

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state, name = 'init_states')(graph)
    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False, mode = 'layer'):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer,),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            if mode == 'layer':
                result.add(tf.keras.layers.LayerNormalization())
            else:
                result.add(tf.keras.layers.BatchNormalization())

        return result 
    

    
    def gcn_convolution(message_dim, receiver_tag):
        # Here we now use a GCN layer 

        regularizer = tf.keras.regularizers.l2(l2_reg_factor)


        return  gcn_conv.GCNConv(
            units = message_dim,
            receiver_tag= receiver_tag,
            activation = "relu",
            use_bias = True,
            kernel_regularizer = regularizer,
            add_self_loops = False,
            edge_weight_feature_name= 'weights',
            degree_normalization= 'in'
        )

    



    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    

    if not dilation:
        n_dilation_layers = 1

    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : (gcn_convolution(message_dim, tfgnn.SOURCE))},
                next_state(next_state_dim, use_layer_normalization)
                )
            }
        )(graph)



    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, context_mode, node_set_name = "frames")(graph)  
    

    logits = tf.keras.layers.Dense(num_classes)(pooled_features)


    model = tf.keras.Model(input_graph, logits)


    return model 


"""
def base_gnn_model_using_gcn_with_residual_blocks(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims=128,
        message_dim=128,
        next_state_dim=128,
        skip_connection_type=None,
        num_classes=35,
        l2_reg_factor=6e-6,
        dropout_rate=0.2,
        use_layer_normalization=True,
        n_message_passing_layers=4,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',
        
        ):
    
    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)
    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()

    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        '''
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        '''


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

                # TODO : try to do base mfcc + its energy, delta + energy, delta-delta + energy
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_state, name='init_states')(graph)
    
    # Define layer building blocks
    def dense(units, use_layer_normalization=False):
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation="relu",
                use_bias=True,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization()) # BatchNormalization ?
        return result 
    
    def gcn_convolution(message_dim, receiver_tag):
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        return gcn_conv.GCNConv(
            units=message_dim,
            receiver_tag=receiver_tag,
            activation="relu",
            use_bias=True,
            kernel_regularizer=regularizer,
            add_self_loops=False,
            edge_weight_feature_name="weights",
            degree_normalization="in"
        )

    # Create a residual block as a custom keras model
    class GCNResidualBlock(tf.keras.Model):
        def __init__(self, message_dim, next_state_dim, use_layer_normalization):
            super().__init__()
            # First GCN layer and state update
            self.gcn1 = gcn_convolution(message_dim, tfgnn.SOURCE)
            self.next_state1 = tfgnn.keras.layers.NextStateFromConcat(
                dense(next_state_dim, use_layer_normalization))
            
            # Graph update layer
            self.graph_update = tfgnn.keras.layers.GraphUpdate(
                node_sets={
                    "frames": tfgnn.keras.layers.NodeSetUpdate(
                        {"connections_0": (self.gcn1)},
                        self.next_state1
                    )
                }
            )
        
        def call(self, inputs):
            # Process the graph through the GCN
            outputs = self.graph_update(inputs)
            
            # Apply the residual connection if needed
            if skip_connection_type == 'sum':
                # Extract the node states from input and output graphs
                input_state = inputs.node_sets["frames"]["hidden_state"]
                output_state = outputs.node_sets["frames"]["hidden_state"]
                
                # Create a new graph with the residual connection
                result = tfgnn.GraphTensor.from_pieces(
                    context=outputs.context,
                    node_sets={
                        "frames": tfgnn.NodeSet.from_fields(
                            sizes=outputs.node_sets["frames"].sizes,
                            features={
                                **outputs.node_sets["frames"].features,
                                "hidden_state": input_state + output_state
                            }
                        )
                    },
                    edge_sets=outputs.edge_sets
                )
                return result
            
            return outputs
    
    # Process graph through residual blocks
    for i in range(n_message_passing_layers):
        # Create and apply a residual block
        block = GCNResidualBlock(
            message_dim=message_dim,
            next_state_dim=next_state_dim,
            use_layer_normalization=use_layer_normalization
        )
        
        # Skip connection in the first layer doesn't make sense
        if i == 0:
            # For the first layer, just apply the GCN without residual
            graph = tfgnn.keras.layers.GraphUpdate(
                node_sets={
                    "frames": tfgnn.keras.layers.NodeSetUpdate(
                        {"connections_0": (gcn_convolution(message_dim, tfgnn.SOURCE))},
                        tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization))
                    )
                }
            )(graph)
        else:
            # For subsequent layers, use the residual block
            graph = block(graph)
    
    # Final pooling and classification
    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, context_mode, node_set_name="frames")(graph)
    
    # Add a final classifier layer
    logits = tf.keras.layers.Dense(num_classes)(pooled_features)
    
    # Create the model
    model = tf.keras.Model(input_graph, logits)
    
    return model
"""


def base_gnn_with_context_node_model_v2(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_residual_next_state = False,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',


        ):
    
    """

    Here, we additionally add a context node ("Master node") to the graph, which is used to aggregate information from all nodes.
    In the Base GNN model, we simply aggregated the final information for the master node representation. Here, we aim to
    make the context node a learnable feature instead of just aggregating information into it.

    Compared to v1 of this model, we only update the context node after the message passing layers in order to minimize number of parameters & multiplies.



    """

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)


    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()
        


    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

              
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
        # TODO : can be implemented if we want 
        pass 


    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

        
        return tfgnn.keras.layers.MakeEmptyFeature()(context)



    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        context_fn = set_initial_context_state, name = 'init_states')(graph) # added initial context state 
    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization())
        return result 
    

    def convolution(message_dim, receiver_tag):
        return tfgnn.keras.layers.SimpleConv(dense(message_dim), "sum", receiver_tag = receiver_tag, receiver_feature= None)
    


    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        

    def next_state_concat(next_state_dim, use_layer_normalization):
        # Needed for the context update
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    

    if not dilation:
        n_dilation_layers = 1
    
    # after the first message passing layer, nodes are (3136,128) and context node is (32,128)
    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : convolution(message_dim, tfgnn.SOURCE)},
                next_state(next_state_dim, use_layer_normalization)
                )
            },

            
        )(graph)



    # Finally, update context node
    graph = tfgnn.keras.layers.GraphUpdate(
            context = tfgnn.keras.layers.ContextUpdate(
                {
                    "frames" : tfgnn.keras.layers.Pool(tfgnn.CONTEXT, context_mode, node_set_name = "frames")
                },
                next_state_concat(next_state_dim, use_layer_normalization)
    )
    )(graph)

    # Get the current context state (has shape (batch_size, 128) , where 128 is the message_passing_dimension)
    # This represents the master node, which is updated in each message passing layer !
    context_state = graph.context.features['hidden_state']

    logits = tf.keras.layers.Dense(num_classes)(context_state)

    model = tf.keras.Model(input_graph, logits)

    return model 

def base_gnn_with_context_node_model(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_residual_next_state = False,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',


        ):
    
    """

    Here, we additionally add a context node ("Master node") to the graph, which is used to aggregate information from all nodes.
    In the Base GNN model, we simply aggregated the final information for the master node representation. Here, we make the context node
    a learnable feature that is updated in each message passing layer by putting its current state and all the nodes sent messages through
    a dense layer.


    """

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)


    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()
        


    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

               
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
      
        pass 


    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

        return tfgnn.keras.layers.MakeEmptyFeature()(context)



    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        context_fn = set_initial_context_state, name = 'init_states')(graph) # added initial context state 
    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization())
        return result 
    

    def convolution(message_dim, receiver_tag):
        return tfgnn.keras.layers.SimpleConv(dense(message_dim), "sum", receiver_tag = receiver_tag, receiver_feature= None)
    


    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        

    def next_state_concat(next_state_dim, use_layer_normalization):
        # Needed for the context update
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    

    if not dilation:
        n_dilation_layers = 1
    
    # after the first message passing layer, nodes are (3136,128) and context node is (32,128)
    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : convolution(message_dim, tfgnn.SOURCE)},
                next_state(next_state_dim, use_layer_normalization)
                )
            },
            # Here we just do an easy implementation : in each message passing layer, take all the current node features
            # and pool them using the mean
            context = tfgnn.keras.layers.ContextUpdate(
                {
                    "frames" : tfgnn.keras.layers.Pool(tfgnn.CONTEXT, context_mode, node_set_name = "frames")
                },
                next_state_concat(next_state_dim, use_layer_normalization)
            
        ))(graph)



    # Get the current context state (has shape (batch_size, 128) , where 128 is the message_passing_dimension)
    # This represents the master node, which is updated in each message passing layer !
    context_state = graph.context.features['hidden_state']

    logits = tf.keras.layers.Dense(num_classes)(context_state)


    
    model = tf.keras.Model(input_graph, logits)


    return model 


def base_gnn_weighted_model(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        use_residual_next_state = False,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',


        ):
    

    """
    
    Base GNN, but now using the edge weights : Therefore, trained with the cosine window/similarity
    adjacency matrix approach.
    
    """
    
    if use_residual_next_state:
        initial_nodes_mfccs_layer_dims = message_dim

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()


    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

             
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
        
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """

        pass 


    def set_initial_context_state():
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """
     
        pass


        

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state, name = 'init_states')(graph)
    

    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False, normalization_type = "normal"):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            if normalization_type == 'normal':
                result.add(tf.keras.layers.LayerNormalization())
            elif normalization_type == 'group':
                result.add(tf.keras.layers.GroupNormalization(message_dim))
        return result 
    


    

    # Message passing with edge weights

    # Define a custom class object for the weighted convolution
    # This class will inherit from tf.keras.layers.AnyToAnyConvolutionBase
    


    class WeightedSumConvolution(tf.keras.layers.Layer):

        def __init__(self, message_dim, receiver_tag):
            super().__init__()
            self.message_dim = message_dim
            self.receiver_tag = receiver_tag
            self.sender_tag = tfgnn.SOURCE if receiver_tag == tfgnn.TARGET else tfgnn.TARGET
            self.dense = dense(units = message_dim, use_layer_normalization = use_layer_normalization)
        
        def call(self, graph, edge_set_name):
            # Get node states
            messages = tfgnn.broadcast_node_to_edges(
                graph,
                edge_set_name,
                self.sender_tag,
                feature_name="hidden_state") # Take the hidden state of the node
            
            # Get edge weights
            weights = graph.edge_sets[edge_set_name].features['weights']
            
            # Apply weights to messages
            weighted_messages = tf.expand_dims(weights, -1) * messages
            
            # Pool messages to target nodes
            pooled_messages = tfgnn.pool_edges_to_node(
                graph,
                edge_set_name,
                self.receiver_tag,
                reduce_type='sum',
                feature_value=weighted_messages)
            
            # Transform pooled messages
            return self.dense(pooled_messages)
            

    

    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    

    if not dilation:
        n_dilation_layers = 1

    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : WeightedSumConvolution(message_dim, tfgnn.TARGET)},
                next_state(next_state_dim, use_layer_normalization)
                )
            }
        )(graph)



    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, context_mode, node_set_name = "frames")(graph)   
    logits = tf.keras.layers.Dense(num_classes)(pooled_features)


    
    model = tf.keras.Model(input_graph, logits)


    return model 


def base_gnn_weighted_with_context_model_v2(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        use_residual_next_state = False,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',


        ):
    

    """
    
    Using the weighted edges + context update only AFTER message passing layers.
   
    
    """
    
    if use_residual_next_state:
        initial_nodes_mfccs_layer_dims = message_dim

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()


    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

               
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
        
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """

        pass 


    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

 
        return tfgnn.keras.layers.MakeEmptyFeature()(context)


        
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        context_fn = set_initial_context_state, name = 'init_states')(graph) # added initial context state 
    

    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False, normalization_type = "normal"):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            if normalization_type == 'normal':
                result.add(tf.keras.layers.LayerNormalization())
            elif normalization_type == 'group':
                result.add(tf.keras.layers.GroupNormalization(message_dim))
        return result 
    


    

    # Message passing with edge weights

    # Define a custom class object for the weighted convolution
    # This class will inherit from tf.keras.layers.AnyToAnyConvolutionBase
    


    class WeightedSumConvolution(tf.keras.layers.Layer):

        def __init__(self, message_dim, receiver_tag):
            super().__init__()
            self.message_dim = message_dim
            self.receiver_tag = receiver_tag
            self.sender_tag = tfgnn.SOURCE if receiver_tag == tfgnn.TARGET else tfgnn.TARGET
            self.dense = dense(units = message_dim, use_layer_normalization = use_layer_normalization)
        
        def call(self, graph, edge_set_name):
            # Get node states
            messages = tfgnn.broadcast_node_to_edges(
                graph,
                edge_set_name,
                self.sender_tag,
                feature_name="hidden_state") # Take the hidden state of the node
            
            # Get edge weights
            weights = graph.edge_sets[edge_set_name].features['weights']
            
            # Apply weights to messages
            weighted_messages = tf.expand_dims(weights, -1) * messages
            
            # Pool messages to target nodes
            pooled_messages = tfgnn.pool_edges_to_node(
                graph,
                edge_set_name,
                self.receiver_tag,
                reduce_type='sum',
                feature_value=weighted_messages)
            
            # Transform pooled messages
            return self.dense(pooled_messages)
            

    

    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        

    def next_state_concat(next_state_dim, use_layer_normalization):
        # Needed for the context update
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    

    if not dilation:
        n_dilation_layers = 1

    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : WeightedSumConvolution(message_dim, tfgnn.TARGET)},
                next_state(next_state_dim, use_layer_normalization)
                )
            }
        )(graph)



    # Finally, update context node
    graph = tfgnn.keras.layers.GraphUpdate(
            context = tfgnn.keras.layers.ContextUpdate(
                {
                    "frames" : tfgnn.keras.layers.Pool(tfgnn.CONTEXT, context_mode, node_set_name = "frames")
                },
                next_state_concat(next_state_dim, use_layer_normalization)
    )
    )(graph)

    # Get the current context state (has shape (batch_size, 128) , where 128 is the message_passing_dimension)
    # This represents the master node, which is updated in each message passing layer !
    context_state = graph.context.features['hidden_state']

    logits = tf.keras.layers.Dense(num_classes)(context_state)

    model = tf.keras.Model(input_graph, logits)

    return model 





def base_gnn_weighted_with_context_model(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        use_residual_next_state = False,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal',


        ):
    

    """
    
    Using the weighted edges + context updates in each message passing layer.
   
    
    """
    
    if use_residual_next_state:
        initial_nodes_mfccs_layer_dims = message_dim

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()


    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

          
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
        
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
 
        pass 


    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

        return tfgnn.keras.layers.MakeEmptyFeature()(context)


        
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        context_fn = set_initial_context_state, name = 'init_states')(graph) # added initial context state 
    

    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False, normalization_type = "normal"):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            if normalization_type == 'normal':
                result.add(tf.keras.layers.LayerNormalization())
            elif normalization_type == 'group':
                result.add(tf.keras.layers.GroupNormalization(message_dim))
        return result 
    


    

    # Message passing with edge weights

    # Define a custom class object for the weighted convolution
    # This class will inherit from tf.keras.layers.AnyToAnyConvolutionBase
    


    class WeightedSumConvolution(tf.keras.layers.Layer):

        def __init__(self, message_dim, receiver_tag):
            super().__init__()
            self.message_dim = message_dim
            self.receiver_tag = receiver_tag
            self.sender_tag = tfgnn.SOURCE if receiver_tag == tfgnn.TARGET else tfgnn.TARGET
            self.dense = dense(units = message_dim, use_layer_normalization = use_layer_normalization)
        
        def call(self, graph, edge_set_name):
            # Get node states
            messages = tfgnn.broadcast_node_to_edges(
                graph,
                edge_set_name,
                self.sender_tag,
                feature_name="hidden_state") # Take the hidden state of the node
            
            # Get edge weights
            weights = graph.edge_sets[edge_set_name].features['weights']
            
            # Apply weights to messages
            weighted_messages = tf.expand_dims(weights, -1) * messages
            
            # Pool messages to target nodes
            pooled_messages = tfgnn.pool_edges_to_node(
                graph,
                edge_set_name,
                self.receiver_tag,
                reduce_type='sum',
                feature_value=weighted_messages)
            
            # Transform pooled messages
            return self.dense(pooled_messages)
            

    

    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        

    def next_state_concat(next_state_dim, use_layer_normalization):
        # Needed for the context update
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    

    if not dilation:
        n_dilation_layers = 1

    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : WeightedSumConvolution(message_dim, tfgnn.TARGET)},
                next_state(next_state_dim, use_layer_normalization)
                )
            },

            context = tfgnn.keras.layers.ContextUpdate(
        {
            "frames" : tfgnn.keras.layers.Pool(tfgnn.CONTEXT, context_mode, node_set_name = "frames")
        },
        next_state_concat(next_state_dim, use_layer_normalization)
        

        ))(graph)



    # Get the current context state (has shape (batch_size, 128) , where 128 is the message_passing_dimension)
    # This represents the master node, which is updated in each message passing layer !
    context_state = graph.context.features['hidden_state']

    logits = tf.keras.layers.Dense(num_classes)(context_state)

    model = tf.keras.Model(input_graph, logits)

    return model 


def GCN_model(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_residual_next_state = False,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,
        context_mode = 'mean',
        initial_state_mfcc_mode = 'normal'
        ):

    """
    Base GCN model + learnable context node in each message passing layer.
    
    """

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()



    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            

            if initial_state_mfcc_mode == 'normal':
                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                    node_set["features"]  # This would be your mfcc_static features
                )
            elif initial_state_mfcc_mode == 'splitted':
                # Split the diff. features such that we can do separate layer learning

                features = node_set["features"]

               
                base_mfccs = features[: , 0:12]
                delta_mfccs = features[: , 12:24]
                delta_delta_mfccs = features[:, 24:36]
                energy_features = features[:, 36:39]

                base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
                delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
                delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
                energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

                # Concatenate the processed features
                combined_features = tf.keras.layers.Concatenate()(
                    [base_processed, delta_processed, delta_delta_processed, energy_processed]
                )
                


                return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            
            elif initial_state_mfcc_mode == 'conv':
                x = node_set["features"]

                x = tf.expand_dims(x, -1)

                conv_out = tf.keras.layers.Conv1D(16, kernel_size = 3, padding="same")(x)

                flattened_out = tf.keras.layers.Flatten()(conv_out)

                return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation = "relu")(flattened_out)
                
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
        

    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """

        return tfgnn.keras.layers.MakeEmptyFeature()(context)



    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        context_fn = set_initial_context_state, name = 'init_states')(graph)
    

    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = "relu",
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            result.add(tf.keras.layers.LayerNormalization())
        return result 
    

    
    def gcn_convolution(message_dim, receiver_tag):
        # Here we now use a GCN layer 
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)


        return  gcn_conv.GCNConv(
            units = message_dim,
            receiver_tag= receiver_tag,
            activation = "relu",
            use_bias = True,
            kernel_regularizer = regularizer,
            add_self_loops = False,
            edge_weight_feature_name= 'weights',
            degree_normalization= 'in'
        )


    def next_state(next_state_dim, use_layer_normalization):
        if not use_residual_next_state:
            return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        else:
            return tfgnn.keras.layers.ResidualNextState(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
        

    def next_state_concat(next_state_dim, use_layer_normalization):
        # Needed for the context update
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    

    if not dilation:
        n_dilation_layers = 1

    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : (gcn_convolution(message_dim, tfgnn.SOURCE))},
                next_state(next_state_dim, use_layer_normalization)
                )
            },

            context = tfgnn.keras.layers.ContextUpdate(
                {
                    "frames" : tfgnn.keras.layers.Pool(tfgnn.CONTEXT, context_mode, node_set_name = "frames")
                },
                next_state_concat(next_state_dim, use_layer_normalization)
            
        ))(graph)



    context_state = graph.context.features['hidden_state']

    logits = tf.keras.layers.Dense(num_classes)(context_state)

    model = tf.keras.Model(input_graph, logits)

    return model 
    


        



def train(model, train_ds, val_ds, test_ds, epochs = 50, batch_size = 32, use_callbacks = True, learning_rate = 0.001):

    """
    Training Function. We utilize a learning rate scheduler as we found it beneficial for our GNN models.
    
    """


    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,# changed patience to 2
            min_lr=1e-10, # changed from 1e-6
            verbose=1
        ),

        # Tensorflow doesn't provide model saving for GNNs, so we save weights in a checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model_weights.h5',  
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,  
            verbose=1)
    ]



    model.compile(
        # legacy due to running on mac m1
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = learning_rate),
        # using sparse categorical bc our labels are encoded as numbers and not one-hot
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()],
       # run_eagerly = True
    )


    if use_callbacks:
        history = model.fit(train_ds, validation_data = val_ds, epochs = epochs, callbacks = callbacks)
    else:
        history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)


    # Evaluate the model
    test_measurements = model.evaluate(test_ds)


    print(f"Test Loss : {test_measurements[0]:.2f},\
          Test Sparse Categorical Accuracy : {test_measurements[1]:.2f}")




    return history



## Hierarchical GNN model
# NOTE : We didn't test this model, since we had a gradient flow issue that we could not solve in time.
'''
def base_gnn_hierarchical_model(
        graph_tensor_specification,
        initial_nodes_mfccs_layer_dims = 64,
        initial_nodes_clusters_layer_dims = 64,
        initial_edges_weights_layer_dims = [16],
        message_dim = 128,
        next_state_dim = 128,
        num_classes = 35,
        l2_reg_factor = 6e-6,
        dropout_rate = 0.2,
        use_layer_normalization = True,
        n_message_passing_layers = 4,
        dilation = False,
        n_dilation_layers = 2,


        ):
    

    """
    
    Initial Node Set : 98 Frames ; Edges there : Using normal methods (e.g. weighted using our cosine window approach)
    Second Node Set : 10 Nodes ; Edges there : Directed edges from all the 98 nodes to the 10 nodes (i.e. just send information in one direction) , where we learn the edge features
                # Idea is that this simulates a clustering approach
    Finally : Send clustered messages to context node, using an attention mechanism (hope is that clustered information is easier "separable" for attention mechanism), while working
    on all 98 nodes with the attention mechanism (as in base GATv2 model) might be not optimal ; also, we have less needed parameters for the attention mechanism (but more therefore
    for the learnt edges between the two node sets)
 
    
    """
    

    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()


    # Define the initial hidden states for the nodes
    def set_initial_node_state(node_set,node_set_name):
        """
        Initialize hidden states for nodes in the graph.
        
        Args:
            node_set: A dictionary containing node features
            node_set_name: The name of the node set (e.g., "frames")
            
        Returns:
            A transformation function applied to the node features
        """


        def dense_inner(units, use_layer_normalization = False, normalization_type = "normal"):
            regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            result = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation = "relu",
                    use_bias = True,
                    kernel_regularizer = regularizer,
                    bias_regularizer = regularizer),
                tf.keras.layers.Dropout(dropout_rate)])
            if use_layer_normalization:
                if normalization_type == 'normal':
                    result.add(tf.keras.layers.LayerNormalization())
                elif normalization_type == 'group':
                    result.add(tf.keras.layers.GroupNormalization(message_dim))
            return result 


        if node_set_name == "frames":


            features = node_set["features"]

            # Split the diff. features such that we can do separate layer learning


     
            base_mfccs = features[: , 0:12]
            delta_mfccs = features[: , 12:24]
            delta_delta_mfccs = features[:, 24:36]
            energy_features = features[:, 36:39]

            base_processed = dense_inner(24, use_layer_normalization=True)(base_mfccs)
            delta_processed = dense_inner(24, use_layer_normalization=True)(delta_mfccs)
            delta_delta_processed = dense_inner(24, use_layer_normalization=True)(delta_delta_mfccs)
            energy_processed = dense_inner(8, use_layer_normalization=True)(energy_features)

            # Concatenate the processed features
            combined_features = tf.keras.layers.Concatenate()(
                [base_processed, delta_processed, delta_delta_processed, energy_processed]
            )
            


            return dense_inner(initial_nodes_mfccs_layer_dims, use_layer_normalization=True)(combined_features)
            

        elif node_set_name == "clusters":

            features = node_set["features"]


            # Possibly other modes (i.e. uniformly values between 0 & 1 for initial etc. ; don't know whats
            # best for our use case here)
            return dense_inner(initial_nodes_clusters_layer_dims, use_layer_normalization= True)(features)

        
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
        
    # Needs to be done such that we can learn on the edge weights 
    def map_edge_features(edge_set, edge_set_name):
        if edge_set_name == "frames_to_clusters":
            return {"weights": tf.expand_dims(edge_set["weights"], axis=-1)}
        return edge_set.features

    graph = tfgnn.keras.layers.MapFeatures(
        edge_sets_fn=map_edge_features
    )(graph)
        
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize edge states, with special handling for frames_to_clusters edges.
        For frames_to_clusters, we use a simple learnable scalar weight (0-1).
        
        Args:
            edge_set: Edge set features
            edge_set_name: Name of the edge set
            
        Returns:
            Transformed edge features
        """
        if edge_set_name == "frames_to_clusters":

            initial_weights = edge_set["weights"]
            
            # Create a learnable parameter for each edge
            # We use a single dense layer without bias and with sigmoid activation
            # to ensure the weights stay between 0 and 1
            learnable_weights = tf.keras.layers.Dense(
                units=1,               # Just one scalar value per edge
                activation="sigmoid",  # Keeps weights between 0-1
                use_bias=False,        # No need for bias
                kernel_initializer="glorot_uniform",  # Good for sigmoid
                kernel_regularizer = tf.keras.regularizers.l2(l2_reg_factor)
            )(initial_weights)
            

            return learnable_weights
        



    def set_initial_context_state(context):
        """
        Initialize hidden state for the context of the graph (i.e. the whole graph)
        
        """


        return tfgnn.keras.layers.MakeEmptyFeature()(context)


        

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn = set_initial_node_state,
        edge_sets_fn= set_initial_edge_state,
        context_fn= set_initial_context_state,
          name = 'init_states')(graph)
    

    
    # Let us now build some basic building blocks for our model
    def dense(units, use_layer_normalization = False, activation = "relu", normalization_type = "normal"):
        """ Dense layer with regularization (L2 & Dropout) & normalization"""
        regularizer = tf.keras.regularizers.l2(l2_reg_factor)
        result = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation = activation,
                use_bias = True,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer),
            tf.keras.layers.Dropout(dropout_rate)])
        if use_layer_normalization:
            if normalization_type == 'normal':
                result.add(tf.keras.layers.LayerNormalization())
            elif normalization_type == 'batch':
                result.add(tf.keras.layers.BatchNormalization(message_dim))
        return result 
    



    def gat_convolution(num_heads, receiver_tag):
        # Here we now use a GAT layer

        regularizer = tf.keras.regularizers.l2(l2_reg_factor)


        return  GATv2Conv(
            num_heads = num_heads,
            per_head_channels = 128, # dimension of vector of output of each head
            heads_merge_type = 'concat', # how to merge the heads
            receiver_tag = receiver_tag, # also possible nodes/edges ; see documentation of function !
            receiver_feature = tfgnn.HIDDEN_STATE,
            sender_node_feature = tfgnn.HIDDEN_STATE,
            sender_edge_feature= None,
            kernel_regularizer= regularizer,


        )
    


    


  
    class WeightedSumConvolution(tf.keras.layers.Layer):

        def __init__(self, message_dim, receiver_tag, sender_edge_feature = tfgnn.HIDDEN_STATE, mode = "frame_to_frame"):
            super().__init__()
            self.message_dim = message_dim
            self.receiver_tag = receiver_tag
            self.sender_tag = tfgnn.SOURCE if receiver_tag == tfgnn.TARGET else tfgnn.TARGET
            self.dense = dense(units = message_dim, use_layer_normalization = use_layer_normalization)
            self.sender_edge_feature = sender_edge_feature 
            self.mode = mode
  
        
        def call(self, graph, edge_set_name):
            # Get node states
            messages = tfgnn.broadcast_node_to_edges(
                graph,
                edge_set_name,
                self.sender_tag,
                feature_name="hidden_state") # Take the hidden state of the node
            
            # Get edge weights
            if self.mode == "frame_to_frame":
                weights = graph.edge_sets[edge_set_name].features['weights']
            
            if self.mode == "frame_to_cluster":
                weights = graph.edge_sets[edge_set_name].features[self.sender_edge_feature]
            
            # Apply weights to messages
            if self.mode == "frame_to_frame":
                weighted_messages = tf.expand_dims(weights, -1) * messages
            
            if self.mode == "frame_to_cluster":
                # Already expanded, as we did the MapFeature for the weights at the beginning (since this is needed such that 
                # we can work with trainable edge weights (or, more specifically, their hidden features))
                weighted_messages = weights * messages
            
            # Pool messages to target nodes
            pooled_messages = tfgnn.pool_edges_to_node(
                graph,
                edge_set_name,
                self.receiver_tag,
                reduce_type='sum',
                feature_value=weighted_messages)
            
 


            # Transform pooled messages
            return self.dense(pooled_messages)
    
    
    def next_state_edges(use_layer_normalization):
        return tfgnn.keras.layers.NextStateFromConcat(dense(1, activation = "sigmoid", use_layer_normalization= False))

    

    def next_state(next_state_dim, use_layer_normalization):
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    
    if not dilation:
        n_dilation_layers = 1

    for i in range(n_message_passing_layers):
        dil_layer_num = i % n_dilation_layers # circular usage of dilated adjacency matrices throughout message passing layers
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {f"connections_{dil_layer_num}" : WeightedSumConvolution(message_dim, tfgnn.TARGET, mode = "frame_to_frame")},
                next_state(next_state_dim, use_layer_normalization)
                ),
            # TODO : this leads to problems... no gradient flow
                "clusters" : tfgnn.keras.layers.NodeSetUpdate(
                    {"frames_to_clusters" : WeightedSumConvolution(message_dim = message_dim, receiver_tag = tfgnn.TARGET, mode = "frame_to_cluster")}, # set to target, such that the sender are the 98 initial nodes!
                next_state(next_state_dim= next_state_dim, use_layer_normalization = use_layer_normalization)
                )
            },

            edge_sets = {
                "frames_to_clusters" : tfgnn.keras.layers.EdgeSetUpdate(
                    next_state = next_state_edges(use_layer_normalization),
                    edge_input_feature = tfgnn.HIDDEN_STATE
                  
                )
            },
                
        )(graph)


    # Finally, apply attention mechanism (here we try outside of message passing, such that we don't have so many parameters)

    graph = tfgnn.keras.layers.GraphUpdate(
        
        context = tfgnn.keras.layers.ContextUpdate(
            {
                "clusters" : gat_convolution(num_heads= 2, receiver_tag = tfgnn.CONTEXT)
            },
            next_state(next_state_dim, use_layer_normalization)
        
    )
    
    )(graph)



    # Get the current context state (has shape (batch_size, 128) , where 128 is the message_passing_dimension)
    # This represents the master node, which is updated in each message passing layer !
    context_state = graph.context.features['hidden_state']

    # Dropout # TODO: like in speechreco paper, see if it works/ m
  #  context_state = tf.keras.layers.Dropout(dropout_rate)(context_state)

    logits = tf.keras.layers.Dense(num_classes)(context_state)


    
    model = tf.keras.Model(input_graph, logits)

    return model
'''