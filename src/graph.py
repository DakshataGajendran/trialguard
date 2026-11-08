import networkx as nx
import pandas as pd

def build_account_graph(df):
    """
    Builds a NetworkX graph based on shared identifiers (device_hash, payment_hash, ip_address)
    and returns cluster size and node degree for each account.
    """
    G = nx.Graph()
    
    for _, row in df.iterrows():
        acc_id = row['account_id']
        G.add_node(acc_id)
        
        # Add edges for shared identifiers
        if pd.notna(row['device_hash']):
            G.add_edge(acc_id, f"dev_{row['device_hash']}")
        if pd.notna(row['payment_hash']):
            G.add_edge(acc_id, f"pay_{row['payment_hash']}")
        if pd.notna(row['ip_address']):
            G.add_edge(acc_id, f"ip_{row['ip_address']}")
            
    # Find connected components to measure cluster sizes
    # We can map each account to the size of its connected component
    cluster_sizes = {}
    node_degrees = dict(G.degree())
    
    for component in nx.connected_components(G):
        comp_size = len(component)
        for node in component:
            if node.startswith('A') or node.startswith('B'):  # Account nodes
                cluster_sizes[node] = comp_size
                
    df = df.copy()
    df['graph_cluster_size'] = df['account_id'].map(cluster_sizes).fillna(1)
    df['graph_node_degree'] = df['account_id'].map(node_degrees).fillna(1)
    
    return df

if __name__ == "__main__":
    train_df = pd.read_csv("data/train_accounts.csv")
    train_graph_df = build_account_graph(train_df)
    print(f"Graph features added! Max cluster size: {train_graph_df['graph_cluster_size'].max()}")