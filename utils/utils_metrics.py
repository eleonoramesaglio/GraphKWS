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





'''
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
           # if not use_layer_normalization:
           # GCN block doesn't use any layer normalization
            next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
            # - With layer normalization
           # else:
           #     next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

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

'''













def splitted_multiplications_helper_new(nodes, initial_nodes_mfccs_layer_dims, use_layer_normalization=False):
    """
    Helper for the splitted initial node encoding multiplication calculation. Fixed issue of layer normalization. 
    """

    # Feature dimensions for each component (fixed)
    base_mfccs_dim = 12       # features[:, 0:12]
    delta_mfccs_dim = 12      # features[:, 12:24]
    delta_delta_mfccs_dim = 12  # features[:, 24:36]
    energy_features_dim = 3   # features[:, 36:39]

    # Output dimensions for each dense_inner processing (fixed)
    base_output_dim = 24
    delta_output_dim = 24
    delta_delta_output_dim = 24
    energy_output_dim = 8

    # Per-branch dense (+ optional LN)
    base_mult = nodes * (base_mfccs_dim * base_output_dim)
    delta_mult = nodes * (delta_mfccs_dim * delta_output_dim)
    delta_delta_mult = nodes * (delta_delta_mfccs_dim * delta_delta_output_dim)
    energy_mult = nodes * (energy_features_dim * energy_output_dim)

    if use_layer_normalization:
        base_mult += nodes * (3 * base_output_dim)
        delta_mult += nodes * (3 * delta_output_dim)
        delta_delta_mult += nodes * (3 * delta_delta_output_dim)
        energy_mult += nodes * (3 * energy_output_dim)

    # After concatenation: 24 + 24 + 24 + 8 = 80
    combined_dim = base_output_dim + delta_output_dim + delta_delta_output_dim + energy_output_dim

    # Final dense (+ optional LN): combined_dim -> initial_nodes_mfccs_layer_dims
    final_mult = nodes * (combined_dim * initial_nodes_mfccs_layer_dims)
    if use_layer_normalization:
        final_mult += nodes * (3 * initial_nodes_mfccs_layer_dims)

    total_mult = base_mult + delta_mult + delta_delta_mult + energy_mult + final_mult
    return total_mult


def calculate_multiplications_new(mode, feature_dim, num_edges, message_dim, next_state_dim, message_layers,
                              reduced = False, k_reduced = 0,
                              num_heads = 2, per_head_channels = 128, use_layer_normalization = True, init_node_enc = 'normal'):
    """
    Calculate the number of multiplications for a given model per sample. Fixed an issue where layer normalization was not used
    for every layer.


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
            # Dense: mfccs -> feature_dim
            num_multiplications = nodes * mfccs * feature_dim
            # + LayerNorm on the output of the dense
            if use_layer_normalization:
                num_multiplications += nodes * (3 * feature_dim)

        else:  # splitted mode
            # Split pipeline (each branch dense + LN, final dense + LN)
            num_multiplications = splitted_multiplications_helper_new(
                nodes,
                feature_dim,
                use_layer_normalization=use_layer_normalization
            )

        # 2. Message passing
        for i in range(message_layers):
            node_dim = feature_dim if i == 0 else next_state_dim

            # 2a. GNN SimpleConv
            # Dense inside SimpleConv: node_dim -> message_dim, for each node
            gnn_conv_multiplications = nodes * node_dim * message_dim
            # + LayerNorm on SimpleConv output
            if use_layer_normalization:
                gnn_conv_multiplications += nodes * (3 * message_dim)

            # 2b. Next state computation
            if not use_layer_normalization:
                next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
            else:
                next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gnn_conv_multiplications + next_state_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    # NOT USED 
    elif mode == 'gnn_weighted_context':

        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper_new(nodes, feature_dim)

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

        # 1) Initial state encoding
        if init_node_enc == 'normal':
            # Dense: mfccs -> feature_dim
            num_multiplications = nodes * mfccs * feature_dim
            # + LayerNorm on the dense output
            if use_layer_normalization:
                num_multiplications += nodes * (3 * feature_dim)

        else:  # splitted mode
            # Split pipeline counts Dense (+ optional LN) for each branch and final dense
            num_multiplications = splitted_multiplications_helper_new(
                nodes,
                feature_dim,
                use_layer_normalization=use_layer_normalization
            )

        # 2) Message passing
        for i in range(message_layers):
            node_dim = feature_dim if i == 0 else next_state_dim

            # 2a) GNN WeightedConv
            # - Edge weights application: |E| × node_dim
            # - Dense after pooling: |V| × node_dim × message_dim
            gnn_conv_multiplications = num_edges * node_dim + nodes * node_dim * message_dim

            # + LayerNorm on the dense output of WeightedConv (dimension = message_dim)
            if use_layer_normalization:
                gnn_conv_multiplications += nodes * (3 * message_dim)

            # 2b) Next state computation
            if not use_layer_normalization:
                next_state_multiplications = nodes * (node_dim + message_dim) * next_state_dim
            else:
                next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gnn_conv_multiplications + next_state_multiplications

        # 3) Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications

        
    elif mode == 'base_gcn':
        # Note: This mode computes the number of multiplications for both base_gnn_model_using_gcn and base_gnn_model_using_gcn_with_residual_blocks.
        # the residual connection is just an addition, so it does not contribute to the number of multiplications.

        # 1) Initial state encoding
        if init_node_enc == 'normal':
            # Dense: mfccs -> feature_dim
            num_multiplications = nodes * mfccs * feature_dim
            # + LayerNorm on the output of the dense
            if use_layer_normalization:
                num_multiplications += nodes * (3 * feature_dim)
        else:  # splitted mode
            # Split pipeline counts Dense + (optional) LN for each branch and final dense
            num_multiplications = splitted_multiplications_helper_new(
                nodes,
                feature_dim,
                use_layer_normalization=use_layer_normalization
            )

        # 2) Message passing (GCN block has NO LN)
        for i in range(message_layers):
            node_dim = feature_dim if i == 0 else next_state_dim

            # 2a) GCN convolution:
            #   - Message passing:        |E| × node_dim
            #   - Degree normalization:   |V|
            #   - Dense after agg:        |V| × node_dim × message_dim
            # (No LayerNorm here by design,i.e. in tensorflow GCN block we couldn't add it (not a hyperparameter))
            gcn_conv_multiplications = (
                num_edges * node_dim
                + nodes
                + nodes * node_dim * message_dim
            )

            # 2b) Next state computation (Dense (+ optional LN))
            if not use_layer_normalization:
                next_state_multiplications = nodes * (node_dim + message_dim) * next_state_dim
            else:
                next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gcn_conv_multiplications + next_state_multiplications

        # 3) Mean pooling
        pooling_multiplications = next_state_dim

        # 4) Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += pooling_multiplications + logits_multiplications

    # NOT USED
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

    # NOT USED
    elif mode == 'gat_gcn':
        # gat_gcn updates the context node (with GAT v2) after each message passing layer
        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper_new(nodes, feature_dim)
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
            # Dense: mfccs -> feature_dim
            num_multiplications = nodes * mfccs * feature_dim
            # + LayerNorm on the dense output
            if use_layer_normalization:
                num_multiplications += nodes * (3 * feature_dim)
        else:  # splitted mode
            # Split pipeline counts Dense + (optional) LN for each branch and final dense
            num_multiplications = splitted_multiplications_helper_new(
                nodes,
                feature_dim,
                use_layer_normalization=use_layer_normalization
            )

        context_dim = next_state_dim

        # 2) Message passing (GCN block has NO LN)
        for i in range(message_layers):
            node_dim = feature_dim if i == 0 else next_state_dim

            # 2a) GCN convolution:
            #   - Message passing:        |E| × node_dim
            #   - Degree normalization:   |V|
            #   - Dense after agg:        |V| × node_dim × message_dim
            # (NO LayerNorm here by design)
            gcn_conv_multiplications = (
                num_edges * node_dim
                + nodes
                + nodes * node_dim * message_dim
            )

            # 2b) Next state for nodes: Dense (+ optional LN)
            if not use_layer_normalization:
                next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
            else:
                next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)

            num_multiplications += gcn_conv_multiplications + next_state_node_multiplications

        # 3) GAT v2 convolution to the context node (applied once after all layers):
        # Linear transformations
        #   - For all nodes and the single context: (2*nodes + 1) * node_dim * per_head_channels * num_heads
        linear_multiplications = (2 * nodes + 1) * node_dim * per_head_channels * num_heads
        # Attention computation and application
        attention_multiplications = 2 * nodes * per_head_channels * num_heads
        gat_conv_multiplications = linear_multiplications + attention_multiplications


        # 4) Next state for context node: Dense (+ optional LN)
        if not use_layer_normalization:
            next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim
        else:
            next_state_context_multiplications = (
                (context_dim + num_heads * per_head_channels) * next_state_dim
                + 3 * next_state_dim
            )

        num_multiplications += gat_conv_multiplications + next_state_context_multiplications

        # 5) Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications

    # NOT USED
    elif mode == 'gcn':
        # gcn updates the context node after each message passing layer by mean pooling the node features and sending them to the context node.
        # This is similar to gat_gcn, but without the attention mechanism (therefore with less parameters and multiplications).
        if init_node_enc == 'normal':
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            num_multiplications = nodes * mfccs * feature_dim
        
        else: # splitted mode 
            # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
            # and then we apply a dropout and layer normalization
            num_multiplications = splitted_multiplications_helper_new(nodes, feature_dim)
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





















### AFTER TESTING VISUALIZATIONS :


def visualization_base_gnn():
    models = [
        'Base GNN',
        '5 Dilation Layers,\nWindow Size 5',
        'Cosine Window,\nWeighted Edges',
        'Residual Next State',
        'Multi-Branch\nNode Encoding',
        'Time Shift/\nBest GNN (64 dim)',
        'Best GNN (64 dim)\n+ SpecAug'
    ]

    accuracies = [85.57, 90.17, 90.21, 90.48, 91.08, 91.13, 91.87]
    std_devs = [0.95, 0.21, 0.11, 0.54, 0.44, 0.22, 0.23]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create error bars
    x_pos = np.arange(len(models))
    ax.errorbar(x_pos, accuracies, yerr=std_devs, fmt='o-', capsize=5, 
                color='lightblue', markersize=3, markerfacecolor='lightblue',
                markeredgecolor='steelblue', markeredgewidth=0.5, 
                ecolor='steelblue', elinewidth=2, linewidth=2, linestyle='-')

    ax.set_xlabel('Model Architecture', fontsize= 15,  fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize= 15,  fontweight='bold')


    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=15)

    # Set y-axis range to focus on the relevant accuracy range
    ax.set_ylim(84, 93)


    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (acc, std) in enumerate(zip(accuracies, std_devs)):
        ax.text(i, acc + std + 0.1, f'{acc:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')


    plt.tight_layout()
    plt.savefig('imgs/Base_GNN_to_Best_GNN.png', dpi=300, bbox_inches='tight')



def visualize_window_sizes_effect():
    window_sizes = [5, 10, 15, 20, 25]
    accuracies = [91.13, 90.61, 90.14, 89.26, 88.54]
    std_devs = [0.22, 0.25, 0.16, 0.08, 0.09]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot with connected dots and error bars
    ax.errorbar(window_sizes, accuracies, yerr=std_devs, fmt='o-', capsize=0, 
                color='steelblue', markersize=8, linewidth=2.5, 
                markerfacecolor='lightblue', markeredgecolor='steelblue', 
                markeredgewidth=2, ecolor='steelblue', elinewidth=2)

    # Add custom dashed lines for error bars with horizontal caps
    for ws, acc, std in zip(window_sizes, accuracies, std_devs):
        # Vertical dashed line
        ax.plot([ws, ws], [acc - std, acc + std], 'steelblue', linewidth=2, linestyle='--')
        # Horizontal caps
        cap_width = 0.3
        ax.plot([ws - cap_width, ws + cap_width], [acc - std, acc - std], 'steelblue', 
                linewidth=2, linestyle='-')
        ax.plot([ws - cap_width, ws + cap_width], [acc + std, acc + std], 'steelblue', 
                linewidth=2, linestyle='-')

    # Customize the plot
    ax.set_xlabel('Window Size', fontsize=15, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=15, fontweight='bold')

    # Set axis ranges
    ax.set_xlim(0, 30)
    ax.set_ylim(87.5, 92)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add value labels on each point (with std dev)
    for ws, acc, std in zip(window_sizes, accuracies, std_devs):
        ax.annotate(f'{acc:.2f}±{std:.2f}', (ws, acc), textcoords="offset points", 
                    xytext=(0, std * 100 + 15), ha='center', fontsize=11, fontweight='bold')

    # Set x-axis ticks to show all window sizes
    ax.set_xticks(window_sizes)
    ax.set_xticklabels(window_sizes, fontsize=12)



    # Add legend
    ax.legend(loc='upper right', fontsize=12)



    plt.tight_layout()
    plt.savefig('imgs/BestGNN_window_size_vs_accuracy.png', dpi=300, bbox_inches='tight')



def visualize_reduced_nodes_effect():
    # Data for each model
    models = [
        {
            'name': 'GCN + Normal Node Encoding',
            'color': 'green',
            'reduced_nodes': [0, 2, 4],
            'accuracies': [85.83, 84.94, 81.71],
            'std_devs': [0.70, 0.45, 0.38],
            'multiplies': ['1.748M', '906k', '493k']
        },
        {
            'name': 'Best GNN (64 dim)',
            'color': 'blue',
            'reduced_nodes': [0, 2, 4],
            'accuracies': [91.13, 89.80, 86.77],
            'std_devs': [0.22, 0.33, 0.51],
            'multiplies': ['6.968M', '3.547M', '1.872M']
        },
        {
            'name': 'GAT-GCN',
            'color': 'orange',
            'reduced_nodes': [0, 2, 4],
            'accuracies': [94.58, 94.08, 92.99],
            'std_devs': [0.11, 0.15, 0.13],
            'multiplies': ['15.114M', '7.664M', '4.015M']
        }
    ]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # X positions for each section (0, 2, 4 repeated 3 times with equal spacing)
    section_spacing = 4 
    x_positions = []
    section_centers = []

    for i in range(3):  # 3 sections
        base_x = i * section_spacing
        section_x = [base_x, base_x + 1, base_x + 2]
        x_positions.extend(section_x)
        section_centers.append(base_x + 1)  # Center of each section for labels

    # Plot data for each model
    for i, model in enumerate(models):
        # Get x positions for this model's section
        start_idx = i * 3
        model_x = x_positions[start_idx:start_idx + 3]
        
        # Plot scatter points
        ax.scatter(model_x, model['accuracies'], color=model['color'], s=100, zorder=3)
        
        # Add error bars with horizontal caps
        for j, (x, acc, std) in enumerate(zip(model_x, model['accuracies'], model['std_devs'])):
            # Vertical line
            ax.plot([x, x], [acc - std, acc + std], color=model['color'], 
                    linewidth=2, linestyle='--', zorder=2)
            # Horizontal caps
            cap_width = 0.1
            ax.plot([x - cap_width, x + cap_width], [acc - std, acc - std], color=model['color'], 
                    linewidth=2, linestyle='-', zorder=2)
            ax.plot([x - cap_width, x + cap_width], [acc + std, acc + std], color=model['color'], 
                    linewidth=2, linestyle='-', zorder=2)
        
        # Add multiplies labels above each point
        for x, acc, std, mult in zip(model_x, model['accuracies'], model['std_devs'], model['multiplies']):
            ax.text(x, acc + std + 0.5, mult, ha='center', va='bottom', 
                    fontsize=15, fontweight='bold', color='black')

    # Customize the plot
    ax.set_xlabel('Number of Reduced Nodes', fontsize=15, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=15, fontweight='bold')


    # Set x-axis ticks and labels
    all_x_ticks = []
    all_x_labels = []
    for i in range(3):  # 3 sections
        base_x = i * section_spacing
        all_x_ticks.extend([base_x, base_x + 1, base_x + 2])
        all_x_labels.extend(['0', '2', '4'])

    ax.set_xticks(all_x_ticks)
    ax.set_xticklabels(all_x_labels, fontsize = 17)

    # Set y-axis limits with some padding
    all_accuracies = [acc for model in models for acc in model['accuracies']]
    all_std_devs = [std for model in models for std in model['std_devs']]
    y_min = min(all_accuracies) - max(all_std_devs) - 2
    y_max = max(all_accuracies) + max(all_std_devs) + 3
    ax.set_ylim(y_min, y_max)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', axis='y')

    # Add vertical separators between sections
    for i in range(1, 3):
        sep_x = i * section_spacing - 0.5
        ax.axvline(x=sep_x, color='gray', linestyle=':', alpha=0.5)

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model['color'], 
                                markersize=10, label=model['name']) for model in models]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()

    plt.savefig('imgs/reduced_nodes_comparison.png', dpi=300, bbox_inches='tight')

  

def main():
    visualization_base_gnn()
    visualize_window_sizes_effect()
    visualize_reduced_nodes_effect()


if __name__ == '__main__':
    main()