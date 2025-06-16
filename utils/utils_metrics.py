import matplotlib.pyplot as plt
import math
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score




# Get predicted and true labels

def get_ys(test_ds, base_model):

    y_pred = []
    y_true = []

    for x, y in test_ds:
        predictions = base_model.predict(x)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(y.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return y_pred, y_true


# Confusion matrix visualization

def visualize_confusion_matrix(y_pred, y_true, idx):
    cm = confusion_matrix(y_pred, y_true, normalize="true")
    plt.figure(figsize=(16, 14))
    class_names = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", 
                   "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree",
                   "two", "up", "visual", "wow", "yes", "zero"]
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.savefig(f'imgs/confusion_matrix_{idx}.png')
 #   plt.show()


# Precision, Recall, F1-score
    
def metrics_evaluation(y_pred, y_true, model_name):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted') 
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Create a DataFrame for easy visualization
    metrics_df = pd.DataFrame({
        'Accuracy' : [accuracy],
        'Precision' : [precision],
        'Recall' : [recall],
        'F1-score' : [f1],
        }, index=[model_name])


    # Display the table
    print(metrics_df)


# Plot training history

def plot_history(history, columns=['loss', 'sparse_categorical_accuracy'], idx = 0):
    """
    Plot training history after model has been trained.
    
    Parameters:
    - history: History object returned by model.fit()
    - columns: List of metrics to plot (default: ['loss', 'sparse_categorical_accuracy'])
    """
    # Create subplots
    if len(columns) > 1:
        fig, axes = plt.subplots(len(columns), 1, figsize=(8, 5*len(columns)))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]  # Make it a list for consistent indexing
    
    for i, column in enumerate(columns):
        if column not in history.history:
            print(f"Warning: '{column}' not found in history. Available metrics: {list(history.history.keys())}")
            continue
            
        ax = axes[i]
        ax.plot(history.history[column], label='training', color='blue', linewidth=1.5)
        
        val_column = 'val_'+column
        if val_column in history.history:
            ax.plot(history.history[val_column], label='validation', color='firebrick', linewidth=1.5)
        
        ax.set_xticks(range(len(history.history[column])))
        ax.set_xticklabels(range(1, len(history.history[column])+1))
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.5)
        ax.set_ylabel(column)
        ax.legend(edgecolor='black', facecolor='linen', fontsize=12, loc='best')

    plt.tight_layout()
    plt.savefig(f"imgs/history_plot_{idx}.png")
  #  plt.show()


# Count edges in adjacency matrix

def count_edges(adjacency_matrix):
    """
    Calculate the number of edges in the graph represented by the adjacency matrix.
    """
    # Count the edges
    num_edges = (tf.math.count_nonzero(adjacency_matrix, dtype=tf.int32) // 2).numpy()  # each edge is counted twice in an undirected graph

    return num_edges



def splitted_multiplications_helper(nodes, initial_nodes_mfccs_layer_dims):
    """
    Calculate multiplications for the splitted MFCC processing mode.
    """
    
    # Feature dimensions for each component (fixed)
    base_mfccs_dim = 12      # features[:, 0:12]
    delta_mfccs_dim = 12     # features[:, 12:24] 
    delta_delta_mfccs_dim = 12  # features[:, 24:36]
    energy_features_dim = 3   # features[:, 36:39]
    
    # Output dimensions for each dense_inner processing (fixed)
    base_output_dim = 24
    delta_output_dim = 24
    delta_delta_output_dim = 24
    energy_output_dim = 8
    
    # Calculate multiplications for each processing branch
    # Each dense_inner contains: Dense layer + Dropout + LayerNorm
    # Only Dense layer contributes to multiplications
    
    base_mult = nodes * base_mfccs_dim * base_output_dim
    delta_mult = nodes * delta_mfccs_dim * delta_output_dim  
    delta_delta_mult = nodes * delta_delta_mfccs_dim * delta_delta_output_dim
    energy_mult = nodes * energy_features_dim * energy_output_dim
    
    # Combined features dimension after concatenation
    combined_dim = base_output_dim + delta_output_dim + delta_delta_output_dim + energy_output_dim
    # combined_dim = 24 + 24 + 24 + 8 = 80
    
    # Final dense_inner processing
    final_mult = nodes * combined_dim * initial_nodes_mfccs_layer_dims
    
    # Total multiplications
    total_mult = base_mult + delta_mult + delta_delta_mult + energy_mult + final_mult

    return total_mult



def calculate_multiplications(mode, feature_dim, num_edges, message_dim, next_state_dim, message_layers,
                              reduced = False, k_reduced = 0,
                              num_heads = 2, per_head_channels = 128, use_layer_normalization = True, init_node_enc = 'normal'):
    """
    Calculate the number of multiplications for a given model per sample

    Args:
        mode (str): The model mode, one of 'gnn', 'gnn_weighted_context', 'gnn_weighted', 'base_gcn', 'gat_v2', 'gat_gcn', 'gat_gcn_v2' or 'gcn'.
        feature_dim (int): The dimension of the node features.
        num_edges (int): The number of edges in the graph.
        message_dim (int): The dimension of the messages.
        next_state_dim (int): The dimension of the next state.
        message_layers (int): The number of message passing layers.
        reduced (bool): Whether to use reduced nodes (default: False).
        k_reduced (int): The reduction factor for nodes if reduced is True (default: 0).
        num_heads (int): Number of attention heads for GAT models (default: 2).
        per_head_channels (int): Number of channels per attention head for GAT models (default: 128).
        use_layer_normalization (bool): Whether to use layer normalization in the model (default: True).
        init_node_enc (str): Initial node encoding method, either 'normal' or 'splitted' (default: 'normal').
    Returns:
        int: The total number of multiplications for the model per sample.

    """
    num_multiplications = 0
    num_classes = 35
    mfccs = 39
    if reduced:
        nodes = math.ceil(98/k_reduced)
    else:
        nodes = 98


    if mode == 'gnn':

        # 1. Initial state encoding:

        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper(nodes, feature_dim)

        # 2. Message passing
        for i in range(message_layers):
            # In the first iteration, we have as many features as our initial encoding of the node features
            if i == 0:
                node_dim = feature_dim
            else:
                # Afte the first iteration, it has next_state_dim many features
                node_dim = next_state_dim

            # 2a. GNN SimpleConv
            # - Aggregation of the messages from the neighboring nodes (no multiplications)
            # - Dense layer inside SimpleConv from node_dim to message dim, for each node
            gnn_conv_multiplications = nodes * node_dim * message_dim

            # 2b. Next state computation
            # - Without layer normalization            
            if not use_layer_normalization:
                next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
            # - With layer normalization (+ 3 * next_state_dim for each node)
            else:
                next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gnn_conv_multiplications + next_state_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    elif mode == 'gnn_weighted_context':

        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper(nodes, feature_dim)

        # 2. Message passing
        for i in range(message_layers):
            if i == 0:
                node_dim = feature_dim
            else:
                node_dim = next_state_dim

            # 2a. GNN WeightedConv using the formula: |E| × node_dim (edge weigths application) + |V| × node_dim × message_dim (dense transformation after pooling)
            gnn_conv_multiplications = num_edges * node_dim + nodes * node_dim * message_dim

            # 2b. Next state computation
            # - Without layer normalization
            if not use_layer_normalization:
                next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
            # - With layer normalization
            else:
                next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gnn_conv_multiplications + next_state_multiplications

        # Context node update : Uses Next State From Concat, i.e. concatenates current context node representation (dimension : next_state_dim, since the nodes
        # were already updated) with the pooled messages from all nodes (dimension : next_state_dim (initialized as such))
        # and puts it into a dense layer that maps it to next_state_dim
        context_node_multiplications = (next_state_dim + next_state_dim) * next_state_dim
        num_multiplications += context_node_multiplications


        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications



    elif mode == 'gnn_weighted':

        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper(nodes, feature_dim)

        # 2. Message passing
        for i in range(message_layers):
            if i == 0:
                node_dim = feature_dim
            else:
                node_dim = next_state_dim

            # 2a. GNN WeightedConv using the formula: |E| × node_dim (edge weigths application) + |V| × node_dim × message_dim (dense transformation after pooling)
            gnn_conv_multiplications = num_edges * node_dim + nodes * node_dim * message_dim

            # 2b. Next state computation
            # - Without layer normalization
            if not use_layer_normalization:
                next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
            # - With layer normalization
            else:
                next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gnn_conv_multiplications + next_state_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications

        
    elif mode == 'base_gcn':
        # Note: This mode computes the number of multiplications for both base_gnn_model_using_gcn and base_gnn_model_using_gcn_with_residual_blocks.
        #       Indeed, the residual connection is just an addition, so it does not contribute to the number of multiplications.
        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper(nodes, feature_dim)
        
        # 2. Message passing
        for i in range(message_layers):
            if i == 0:
                node_dim = feature_dim
            else:
                node_dim = next_state_dim

            # 2a. GCN convolution using the formula: |E| × node_dim + |V| + |V| × node_dim × message_dim
            # - Message passing step: |E| × node_dim 
            # - Normalization step: |V| (multiplying each node for 1/degree(node))
            # - Dense transformation step: |V| × node_dim × message_dim
            gcn_conv_multiplications = num_edges * node_dim + nodes + nodes * node_dim * message_dim

            # 2b. Next state computation
            # - Without layer normalization
            if not use_layer_normalization:
                next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
            # - With layer normalization
            else:
                next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gcn_conv_multiplications + next_state_multiplications

        # 3. Mean pooling
        pooling_multiplications = next_state_dim

        # 4. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += pooling_multiplications + logits_multiplications


    elif mode == 'gat_v2':

        # 1. Initial state encoding
        # 1a. Conv1D: we apply a 1D convolution to the input features (mfccs) to get the initial node features (16 filters of size 3)
        conv1D_multiplications = nodes * mfccs * 16 * 3
        # 1b. Dense: from the conv1D output to the feature_dim embedding
        dense_multiplications = nodes * (mfccs * 16) * feature_dim
        num_multiplications = conv1D_multiplications + dense_multiplications
        # 1c. Context node (empty) initialization
        context_dim = next_state_dim

        # 2. Message passing
        for i in range(message_layers):
            if i == 0:
                node_dim = feature_dim
            else:
                node_dim = next_state_dim

            # 2a. GNN WeightedConv for nodes updates
            gnn_conv_multiplications = num_edges * node_dim + nodes * node_dim * message_dim

            # 2b. GAT v2 convolution to context node:
            # We are calculating the attention scores between all our nodes and the context node. 

            # a) Linear transformations: Query, Key and Value matrices (Q, K, W)
            #    QUERY: Generated from the single target token () - "What information do I need?"
            #    -> Query transformation for context node: 1 × node_dim × per_head_channels × num_heads
            #    KEY: Generated from all source tokens () - "What information can we provide?"
            #    -> Key transformation for frame nodes: |V| × node_dim × per_head_channels × num_heads
            #    VALUE: Generated from all source tokens () - "Here's our actual information"
            #    -> Value transformation for frame nodes:  |V| × node_dim × per_head_channels × num_heads
            #    
            #    Q : 1 × per_head_channels
            #    K : |V| × per_head_channels
            #    W : |V| × per_head_channels

            # Overall (2|V|+1)* node_dim × per_head_channels × num_heads
            linear_multiplications = (2 * nodes + 1) * node_dim * per_head_channels * num_heads

            # b) Attention scores computation and application
            # - Computation (C = Q × K^T for each head): |V| × per_head_channels × num_heads
            # - Application (C × W for each head): |V| × per_head_channels × num_heads

            # Overall 2 * |V| × per_head_channels × num_heads
            attention_multiplications = 2 * nodes * per_head_channels * num_heads

            gat_conv_multiplications = linear_multiplications + attention_multiplications

            # 2c. Next state computation:
            # - Without layer normalization
            if not use_layer_normalization:
                # Nodes:
                next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
                # Context node: concatenates the heads result (num_heads * per_head channels) to the node features and then applies a dense layer
                next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim
            # - With layer normalization
            else:
                # Nodes:
                next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
                # Context node:
                next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim + 3 * next_state_dim

            num_multiplications += gnn_conv_multiplications + gat_conv_multiplications + next_state_node_multiplications + next_state_context_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    elif mode == 'gat_gcn':
        # gat_gcn updates the context node (with GAT v2) after each message passing layer
        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper(nodes, feature_dim)
        context_dim = next_state_dim

        # 2. Message passing
        for i in range(message_layers):
            if i == 0:
                node_dim = feature_dim
            else:
                node_dim = next_state_dim

            # 2a. GCN convolution using the formula: |E| × node_dim + |V| + |V| × node_dim × message_dim
            gcn_conv_multiplications = num_edges * node_dim + nodes + nodes * node_dim * message_dim

            # 2b. GAT v2 convolution to context node:
            # Linear transformations
            linear_multiplications = (2 * nodes + 1) * node_dim * per_head_channels * num_heads
            # Attention computation and application
            attention_multiplications = 2 * nodes * per_head_channels * num_heads
            gat_conv_multiplications = linear_multiplications + attention_multiplications

            # 2c. Next state computation
            # - Without layer normalization
            if not use_layer_normalization:
                # Nodes:
                next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
                # Context node:
                next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim
            # - With layer normalization
            else:
                # Nodes:
                next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
                # Context node:
                next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim + 3 * next_state_dim

            num_multiplications += gcn_conv_multiplications + gat_conv_multiplications + next_state_node_multiplications + next_state_context_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    elif mode == 'gat_gcn_v2':
        # gat_gcn_v2 only updates the context node (with GAT v2) AFTER all the message passing layers (so always just once)
        # -> less parameters & multiplications compared to the gat_gcn model
        # 1. Initial state encoding
        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper(nodes, feature_dim)

        context_dim = next_state_dim

        # 2. Message passing
        for i in range(message_layers):
            if i == 0:
                node_dim = feature_dim
            else:
                node_dim = next_state_dim
                
            # 2a. GCN convolution using the formula: |E| × node_dim + |V| + |V| × node_dim × message_dim
            gcn_conv_multiplications = num_edges * node_dim + nodes + nodes * node_dim * message_dim

            # 2b. Next state computation for nodes
            # - Without layer normalization
            if not use_layer_normalization:
                next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
            # - With layer normalization
            else:
                next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gcn_conv_multiplications + next_state_node_multiplications

        # 3. GAT v2 convolution to context node:
        # Linear transformations
        linear_multiplications = (2 * nodes + 1) * node_dim * per_head_channels * num_heads
        # Attention computation and application
        attention_multiplications = 2 * nodes * per_head_channels * num_heads
        gat_conv_multiplications = linear_multiplications + attention_multiplications

        # 4. Next state computation for context node:
        # - Without layer normalization
        if not use_layer_normalization:
            next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim
        # - With layer normalization (+ 3 * next_state_dim for each node)
        else:
            next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim + 3 * next_state_dim

        num_multiplications += gat_conv_multiplications + next_state_context_multiplications

        # 5. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    elif mode == 'gcn':
        # gcn updates the context node after each message passing layer by mean pooling the node features and sending them to the context node.
        # This is similar to gat_gcn, but without the attention mechanism (therefore with less parameters and multiplications).
        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper(nodes, feature_dim)
        context_dim = next_state_dim

        # 2. Message passing
        for i in range(message_layers):
            if i == 0:
                node_dim = feature_dim
            else:
                node_dim = next_state_dim
            
            # 2a. GCN convolution using the formula: |E| × node_dim + |V| + |V| × node_dim × message_dim
            gcn_conv_multiplications = num_edges * node_dim + nodes + nodes * node_dim * message_dim

            # 2b. Context node update with mean pooling
            context_pooling_multiplications = next_state_dim

            # 2c. Next state computation
            # - Without layer normalization
            if not use_layer_normalization:
                # Nodes:
                next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
                # Context node:
                next_state_context_multiplications = (context_dim + next_state_dim) * next_state_dim   # concatenates current context + pooled nodes
            # - With layer normalization
            else:
                # Nodes:
                next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
                # Context node:
                next_state_context_multiplications = (context_dim + next_state_dim) * next_state_dim + 3 * next_state_dim

            num_multiplications += gcn_conv_multiplications + context_pooling_multiplications + next_state_node_multiplications + next_state_context_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'gnn', 'gnn_weighted', 'base_gcn', 'gat_v2', 'gat_gcn', 'gat_gcn_v2' and 'gcn'.")

        
    return num_multiplications
