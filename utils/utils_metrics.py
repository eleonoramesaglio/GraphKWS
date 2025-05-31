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


# TODO: add a function to compute the number of edges in the graph first

def count_edges(adjacency_matrix):
    """
    Calculate the number of edges in the graph represented by the adjacency matrix.
    """
    # Count the edges
    num_edges = (tf.math.count_nonzero(adjacency_matrix, dtype=tf.int32) // 2).numpy()  # each edge is counted twice in an undirected graph

    return num_edges


def calculate_multiplications(mode, feature_dim, num_edges, message_dim, next_state_dim, message_layers,
                              reduced = False, k_reduced = 0,
                              num_heads = 2, per_head_channels = 128, use_layer_normalization = True,
                              residual_next_state = False):
    """
    Calculate the number of multiplications for a given model per sample

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
        # For each node, we encode the features using a dense layer that maps from mfccs to feature_dim
        num_multiplications = nodes * mfccs * feature_dim

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

            # NextStateFromConcat
            if not residual_next_state:
                # - Without layer normalization            
                if not use_layer_normalization:
                    next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
                # - With layer normalization (+ 3 * next_state_dim for each node)
                else:
                    next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
            # ResidualNextState
            # (we don't concatenate the message dim to the node features, but add them to the next state)
            else:
                # - Without layer normalization            
                if not use_layer_normalization:
                    next_state_multiplications = next_state_dim * next_state_dim * nodes
                # - With layer normalization:
                else:
                    next_state_multiplications = nodes * (next_state_dim * next_state_dim + 3 * next_state_dim)

            num_multiplications += gnn_conv_multiplications + next_state_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    elif mode == 'gnn_weighted':

        # 1. Initial state encoding
        num_multiplications = nodes * mfccs * feature_dim

        # 2. Message passing
        for i in range(message_layers):
            if i == 0:
                node_dim = feature_dim
            else:
                node_dim = next_state_dim

            # 2a. GNN WeightedConv using the formula: |E| × node_dim (edge weigths application) + |V| × node_dim × message_dim (dense transformation after pooling)
            gnn_conv_multiplications = num_edges * node_dim + nodes * node_dim * message_dim

            # 2b. Next state computation

            # NextStateFromConcat
            if not residual_next_state:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
                # - With layer normalization
                else:
                    next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
                num_multiplications += gnn_conv_multiplications + next_state_multiplications
            # ResidualNextState
            else:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_multiplications = next_state_dim * next_state_dim * nodes
                # - With layer normalization
                else:
                    next_state_multiplications = nodes * (next_state_dim * next_state_dim + 3 * next_state_dim)

                num_multiplications += gnn_conv_multiplications + next_state_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications

        
    elif mode == 'base_gcn':

        # 1. Initial state encoding
        num_multiplications = nodes * mfccs * feature_dim
        
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
            # NextStateFromConcat
            if not residual_next_state:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_multiplications = (node_dim + message_dim) * next_state_dim * nodes
                # - With layer normalization
                else:
                    next_state_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
            # ResidualNextState
            else: 
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_multiplications = next_state_dim * next_state_dim * nodes
                # - With layer normalization
                else:
                    next_state_multiplications = nodes * (next_state_dim * next_state_dim + 3 * next_state_dim)

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

            # NODES:
            # NextStateFromConcat
            if not residual_next_state:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
                # - With layer normalization
                else:
                    next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
            # ResidualNextState
            else:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_node_multiplications = nodes * next_state_dim * next_state_dim
                # - With layer normalization
                else:
                    next_state_node_multiplications = nodes * (next_state_dim * next_state_dim + 3 * next_state_dim)

            # CONTEXT NODE (NextStateFromConcat):
            # - Without layer normalization
            if not use_layer_normalization:
                    # Concatenates the heads result (num_heads * per_head channels) to the node features and then applies a dense layer
                    next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim
            # - With layer normalization
            else:
                next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim + 3 * next_state_dim

            num_multiplications += gnn_conv_multiplications + gat_conv_multiplications + next_state_node_multiplications + next_state_context_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    elif mode == 'gat_gcn':
        # gat_gcn updates the context node (with GAT v2) after each message passing layer
        # 1. Initial state encoding
        num_multiplications = nodes * mfccs * feature_dim
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

            # NODES:
            # NextStateFromConcat
            if not residual_next_state:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
                # - With layer normalization
                else:
                    next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
            # ResidualNextState
            else:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_node_multiplications = nodes * next_state_dim * next_state_dim
                # - With layer normalization
                else:
                    next_state_node_multiplications = nodes * (next_state_dim * next_state_dim + 3 * next_state_dim)

            # CONTEXT NODE (NextStateFromConcat):
            # - Without layer normalization
            if not use_layer_normalization:
                next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim
            # - With layer normalization
            else:
                next_state_context_multiplications = (context_dim + num_heads * per_head_channels) * next_state_dim + 3 * next_state_dim

            num_multiplications += gcn_conv_multiplications + gat_conv_multiplications + next_state_node_multiplications + next_state_context_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


    elif mode == 'gat_gcn_v2':
        # gat_gcn_v2 only updates the context node (with GAT v2) AFTER all the message passing layers (so always just once)
        # -> less parameters & multiplications compared to the gat_gcn model
        # 1. Initial state encoding
        num_multiplications = nodes * mfccs * feature_dim
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
            # NextStateFromConcat
            if not residual_next_state:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
                # - With layer normalization
                else:
                    next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
            # ResidualNextState
            else:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_node_multiplications = nodes * next_state_dim * next_state_dim
                # - With layer normalization
                else:
                    next_state_node_multiplications = nodes * (next_state_dim * next_state_dim + 3 * next_state_dim)

            num_multiplications += gcn_conv_multiplications + next_state_node_multiplications

        # 3. GAT v2 convolution to context node:
        # Linear transformations
        linear_multiplications = (2 * nodes + 1) * node_dim * per_head_channels * num_heads
        # Attention computation and application
        attention_multiplications = 2 * nodes * per_head_channels * num_heads
        gat_conv_multiplications = linear_multiplications + attention_multiplications

        # 4. Next state computation for context node (NextStateFromConcat):
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
        # 1. Initial state encoding
        num_multiplications = nodes * mfccs * feature_dim
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

            # NODES:
            # NextStateFromConcat
            if not residual_next_state:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_node_multiplications = nodes * (node_dim + message_dim) * next_state_dim
                # - With layer normalization
                else:
                    next_state_node_multiplications = nodes * ((node_dim + message_dim) * next_state_dim + 3 * next_state_dim)
            # ResidualNextState
            else:
                # - Without layer normalization
                if not use_layer_normalization:
                    next_state_node_multiplications = nodes * next_state_dim * next_state_dim
                # - With layer normalization
                else:
                    next_state_node_multiplications = nodes * (next_state_dim * next_state_dim + 3 * next_state_dim)
            
            # CONTEXT NODE (NextStateFromConcat):
            # - Without layer normalization
            if not use_layer_normalization:
                next_state_context_multiplications = (context_dim + next_state_dim) * next_state_dim   # concatenates current context + pooled nodes
            # - With layer normalization
            else:
                next_state_context_multiplications = (context_dim + next_state_dim) * next_state_dim + 3 * next_state_dim

            num_multiplications += gcn_conv_multiplications + context_pooling_multiplications + next_state_node_multiplications + next_state_context_multiplications

        # 3. Logits
        logits_multiplications = next_state_dim * num_classes
        num_multiplications += logits_multiplications


        
    return num_multiplications
