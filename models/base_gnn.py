import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np 




def mfccs_to_graph_tensors_for_dataset(mfcc, adjacency_matrix, label):

    # Print shapes to debug
  #  tf.print("MFCC shape:", tf.shape(mfcc))
  #  tf.print("Adjacency shape:", tf.shape(adjacency_matrix))
  #  tf.print("Label shape:", tf.shape(label))


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


        ):
    
    """

    Base GNN Model :

    - We have n_frames many nodes and each pack the MFCCs as features 
    - The adjacency matrix is solely 0 and 1

    graph_tensor_specification : the "description" of the input graph 
    initial_nodes_mfccs_layer_dims, initial_edges_weights_layer_dims : the initial dimensions for encoding of features
    message_dim, next_state_dim : dimensions for the message passing algorithm

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
        if node_set_name == "frames":
            # Apply a dense layer to transform MFCC features into hidden states
            # Instead of just one dense layer , we can also directly use dropout etc. here (if we wish so) 
            return tf.keras.layers.Dense(initial_nodes_mfccs_layer_dims, activation="relu")(
                node_set["features"]  # This would be your mfcc_static features
            )
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
    # TODO: When we will have convolution_with_weights, this function will be maybe not needed:
    # If we have a unweighted adjacency matrix (window case), the function convolution_with_weights will return the same result as the normal convolution
    # If instead we have a weighted adjacency matrix, convolution_with_weights implements a sort of "attention mechanism", giving more importance
    # to the edges with higher weights.
    def convolution(message_dim, receiver_tag):
        return tfgnn.keras.layers.SimpleConv(dense(message_dim), "sum", receiver_tag = receiver_tag)
    
    # Function: AGGREGATION
    # The convolution function is used to AGGREGATE messages from the neighbors of a node to update its state.
    # There are two functions deciding the type of aggregation: reduce_type and combine_type.
    # The reduce_type specifies how to combine the messages from the neighbors (e.g., sum, mean, max).
    # The combine_type is instead used when there are multiple messages (deriving from multiple features) from the same neighbor.
    # It decides how to aggregate these messages (usually they are concatenated) before passing them to the receiver node.
    # The receiver_tag specifies the node that will receive the aggregated messages.
    
    #TODO: With edge weights
    def convolution_with_weights(message_dim, receiver_tag):
        pass


    def next_state(next_state_dim, use_layer_normalization):
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    
    #TODO: define other possibilities for the next state
    # Note: maybe we do not need to define a convolution_with_weights function, but in the aggregation function we stack the node features in a matrix and then
    # in the next_state function we multiply such matrix by the weights of the edges. This is equivalent to a convolution with weights.

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

    for i in range(n_message_passing_layers):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {"connections" : convolution(message_dim, tfgnn.SOURCE)},
                next_state(next_state_dim, use_layer_normalization)
                )
            }
        )(graph)


    # Read out the hidden state of the root node of EACH COMPONENT in the graph, using tfgnn.keras.layer.ReadoutFirstNode
    # I think this is used with graph sampling, but maybe could be helpful for if we get clusters of graphs ?
    # I am a bit unsure

    # Graph Classification 
    # TODO : add the context node such that it is also learnt during diff message passings etc ?
    # TODO : look this up !
    # TODO : not needed to do , but could be helpful ? right now we just aggregate all the node (?) features
    # TODO : in the end, maybe better to learn over all epochs

    # Context node (/master node) is the "global" node of the graph, which is used to aggregate information from all nodes

    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name = "frames")(graph)   # maybe mean is not the best choice, consider also sum/max
    logits = tf.keras.layers.Dense(num_classes)(pooled_features)


    
    model = tf.keras.Model(input_graph, logits)


    return model 



def train(model, train_ds, val_ds, epochs = 50, batch_size = 32, use_callbacks = True):

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
    ]


    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        # using sparse categorical bc our labels are encoded as numbers and not one-hot
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(),
                   tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)]
    )


    if use_callbacks:
        history = model.fit(train_ds, validation_data = val_ds, epochs = epochs, callbacks = callbacks)
    else:
        history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)

    return history 


def eval_test(model, test_ds):
    test_loss, test_measurements = model.evaluate(test_ds)


    print(test_measurements)


