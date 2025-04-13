# TODO 
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
        n_message_passing_layers = 4,


        ):
    
    """

    # TODO 
    In this approach, instead of using SimpleConv() for the message passing, we use GCN layers!
    Note that this only works with homogeneous graphs ; therefore, we use our cosine window
    approach as it is homogeneous and use weighted edges 

    """


    # Input is the graph structure 
    input_graph = tf.keras.layers.Input(type_spec = graph_tensor_specification)

    # Convert to scalar GraphTensor
    graph = tfgnn.keras.layers.MapFeatures()(input_graph)

    is_batched = (graph.spec.rank == 1)

    if is_batched:
        batch_size = graph.shape[0]
        graph = graph.merge_batch_to_components()



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
    

    
    def gcn_convolution(message_dim, receiver_tag):
        # Here we now use a GCN layer 
        # TODO : don't understand how to add dropout ; I think
        # we would need to add it into the GCNConv class itself, since
        # we are not calling a keras layer here, but the whole class 
        # (i.e. we cannot use the sequential function like normally)
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

    


    # TODO : we can design our own next state function !
    def next_state(next_state_dim, use_layer_normalization):
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    


    for i in range(n_message_passing_layers):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {"connections" : (gcn_convolution(message_dim, tfgnn.SOURCE))},
                next_state(next_state_dim, use_layer_normalization)
                )
            }
        )(graph)



    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name = "frames")(graph)   # maybe mean is not the best choice, consider also sum/max
    logits = tf.keras.layers.Dense(num_classes)(pooled_features)


    
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


        ):
    

    """
    
    Using the weighted edges + also a new initial node state encoding ;
    in the end use pooling with max for context node
    
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
            
        else:
            # Handle any other node types
            raise ValueError(f"Unknown node set: {node_set_name}")
        
            
    def set_initial_edge_state(edge_set, edge_set_name):
        """
        Initialize hidden states for edges in the graph
        
        
        """
        # TODO : I need it to be able to use the weights of the edges in the convolution_with_weights function
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
            
    

    #TODO: Else we can try to define a reduce_type(messages, adjacency_matrix) function that gives back the weighted sum of the messages with the edge weights
    # We just need to access the node index and collect the neighbors of the node and then we can multiply the messages with the weights of the edges
    




    def next_state(next_state_dim, use_layer_normalization):
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    


    for i in range(n_message_passing_layers):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {"connections" : WeightedSumConvolution(message_dim, tfgnn.TARGET)},
                next_state(next_state_dim, use_layer_normalization)
                )
            }
        )(graph)



    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "max", node_set_name = "frames")(graph)   
    logits = tf.keras.layers.Dense(num_classes)(pooled_features)


    
    model = tf.keras.Model(input_graph, logits)


    return model 

