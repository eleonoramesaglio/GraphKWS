# BASE GNN WEIGHTED MODEL

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
    
    

    
    # Message passing with edge weights

    def WeightedConv(message_dim, receiver_tag, reduce_type):

        # Define a custom class object for the weighted convolution
        # This class will inherit from tf.keras.layers.AnyToAnyConvolutionBase
        
        class WeightedConv(tf.keras.layers.AnyToAnyConvolutionBase):

            def __init__(self, message_fn, **kwargs):
                super().__init__(**kwargs)
                self._message_fn = message_fn

            def get_config(self):
                return dict(units=self._message_fn.units, **super().get_config())
            
            # This function is called to compute the messages and aggregate them
            def convolve(
                    self, *,
                    sender_node_input, sender_edge_input,
                    broadcast_from_sender_node, pool_to_receiver):
                
                # Initialize
                inputs = []
                # Store the node features of the sender node
                if sender_node_input is not None:
                    inputs.append(broadcast_from_sender_node(sender_node_input))
    
                # Get the messages
                messages = self._message_fn(tf.concat(inputs, axis=-1))
    
                # Extract edge weights from sender_edge_input
                if sender_edge_input is not None:
                    edge_weights = sender_edge_input
      
                    # Apply weights to messages
                    weighted_messages = messages * edge_weights
      
                    # Return weighted sum to receiver nodes
                    return pool_to_receiver(weighted_messages, reduce_type=reduce_type)
                
                else:
                    # Fallback to regular sum if no edge weights are provided
                    return pool_to_receiver(messages, reduce_type=reduce_type)
            
                
        return WeightedConv()
    


    
    """
    AS REFERENCE: SimpleConv class from tfgnn

    class SimpleConv(tf.keras.layers.AnyToAnyConvolutionBase):
    
    def __init__(
      self,
      message_fn: tf.keras.layers.Layer,
      reduce_type: str = "sum",
      *,
      combine_type: str = "concat",
      receiver_tag: const.IncidentNodeTag = const.TARGET,
      receiver_feature: Optional[const.FieldName] = const.HIDDEN_STATE,
      sender_node_feature: Optional[
          const.FieldName] = const.HIDDEN_STATE,
      sender_edge_feature: Optional[const.FieldName] = None,
      **kwargs):
    super().__init__(
        receiver_tag=receiver_tag,
        receiver_feature=receiver_feature,
        sender_node_feature=sender_node_feature,
        sender_edge_feature=sender_edge_feature,
        **kwargs)

    self._message_fn = message_fn
    self._reduce_type = reduce_type
    self._combine_type = combine_type

  def get_config(self):
    return dict(
        message_fn=self._message_fn,
        reduce_type=self._reduce_type,
        combine_type=self._combine_type,
        **super().get_config())

  def convolve(self, *,
               sender_node_input: Optional[tf.Tensor],
               sender_edge_input: Optional[tf.Tensor],
               receiver_input: Optional[tf.Tensor],
               broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
               broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
               pool_to_receiver: Callable[..., tf.Tensor],
               extra_receiver_ops: Any = None,
               training: bool) -> tf.Tensor:
    assert extra_receiver_ops is None, "Internal error: bad super().__init__()"
    # Collect inputs, suitably broadcast.
    inputs = []
    if sender_edge_input is not None:
      inputs.append(sender_edge_input)
    if sender_node_input is not None:
      inputs.append(broadcast_from_sender_node(sender_node_input))
    if receiver_input is not None:
      inputs.append(broadcast_from_receiver(receiver_input))
    # Combine inputs.
    combined_input = ops.combine_values(inputs, self._combine_type)

    # Compute the result.
    messages = self._message_fn(combined_input)
    pooled_messages = pool_to_receiver(messages, reduce_type=self._reduce_type)
    return pooled_messages

    """


   
    def convolution_with_weights(message_dim, receiver_tag):
        return tfgnn.keras.layers.WeightedConv(dense(message_dim), reduce_type="sum", receiver_tag = receiver_tag)
    

    #TODO: Else we can try to define a reduce_type(messages, adjacency_matrix) function that gives back the weighted sum of the messages with the edge weights
    # We just need to access the node index and collect the neighbors of the node and then we can multiply the messages with the weights of the edges
    




    def next_state(next_state_dim, use_layer_normalization):
        return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, use_layer_normalization=use_layer_normalization))
    


    for i in range(n_message_passing_layers):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets = {
                "frames" : tfgnn.keras.layers.NodeSetUpdate(
                    {"connections" : convolution_with_weights(message_dim, tfgnn.SOURCE)},
                next_state(next_state_dim, use_layer_normalization)
                )
            }
        )(graph)



    pooled_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "max", node_set_name = "frames")(graph)   
    logits = tf.keras.layers.Dense(num_classes)(pooled_features)


    
    model = tf.keras.Model(input_graph, logits)


    return model 
