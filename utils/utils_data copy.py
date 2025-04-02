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

    # Add nodes from each node set
    for node_set_name, node_set in node_sets.items():
        # Get the number of nodes in this node set
        num_nodes = tf.reduce_sum(node_set.sizes).numpy()
        
        # Add nodes
        for i in range(num_nodes):
            # Add node to graph with its set name as an attribute
            G.add_node(i, node_set=node_set_name)
    
    # Extract edge sets
    edge_sets = graph_tensor.edge_sets
    
    # Add edges from each edge set
    for edge_set_name, edge_set in edge_sets.items():
        # Get source and target node indices
        source_indices = edge_set.adjacency.source.numpy()
        target_indices = edge_set.adjacency.target.numpy()
        
        # Add edges
        for i in range(len(source_indices)):
            source_idx = int(source_indices[i])
            target_idx = int(target_indices[i])
            
            # Add edge to graph
            G.add_edge(source_idx, target_idx, edge_set=edge_set_name)
    
    return G
    




def visualize_graph(G, pos=None):
    """
    
    Parameters:
    - G: NetworkX graph
    - pos: optional pre-computed positions """

    plt.figure(figsize=(12, 8))
    
    # If no position provided, use spring layout
    if pos is None:
        pos = nx.spring_layout(G, seed = 1987)
    # Spectral layout may reveal underlying structural properties of the graph, but it's more computationally expensive than other layouts.
    # Alternative: spring_layout() for a force-directed layout (attractive forces between connected nodes, repulsive forces between all nodes)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                            node_size=50)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    plt.title("Speech Feature Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
