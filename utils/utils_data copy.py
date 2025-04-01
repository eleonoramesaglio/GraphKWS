
def visualize_graph(G, pos=None):
    """
    Visualize the graph
    
    Parameters:
    - G: NetworkX graph
    - pos: optional pre-computed positions
    """
    plt.figure(figsize=(12, 8))
    
    # If no position provided, use spring layout
    if pos is None:
        pos = nx.spectral_layout(G)
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

