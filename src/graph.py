
import networkx as nx
import pandas as pd


def build_account_graph(df):
    """
    Build an account relationship graph.

    Detects:
    - Shared devices
    - Shared payment identifiers
    - Shared IP addresses

    Returns graph features for every account.
    """

    df = df.copy()

    required_columns = [
        "account_id",
        "device_hash",
        "payment_hash",
        "ip_address"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing graph columns: {missing}"
        )

    graph = nx.Graph()

    # --------------------------------------------------
    # 1. BUILD ACCOUNT AND IDENTIFIER NODES
    # --------------------------------------------------

    for _, row in df.iterrows():

        if pd.isna(row["account_id"]):
            continue

        account_id = str(row["account_id"]).strip()

        if not account_id:
            continue

        account_node = ("account", account_id)

        graph.add_node(
            account_node,
            node_type="account"
        )

        identifiers = [
            ("device", row["device_hash"]),
            ("payment", row["payment_hash"]),
            ("ip", row["ip_address"])
        ]

        for identifier_type, value in identifiers:

            if pd.isna(value):
                continue

            value = str(value).strip()

            if not value:
                continue

            identifier_node = (
                identifier_type,
                value
            )

            graph.add_node(
                identifier_node,
                node_type="identifier"
            )

            graph.add_edge(
                account_node,
                identifier_node
            )

    # --------------------------------------------------
    # 2. CALCULATE ACCOUNT RELATIONSHIPS
    # --------------------------------------------------

    cluster_sizes = {}
    account_degrees = {}

    shared_device_counts = {}
    shared_payment_counts = {}
    shared_ip_counts = {}

    for component in nx.connected_components(graph):

        component_accounts = [
            node for node in component
            if node[0] == "account"
        ]

        account_count = len(component_accounts)

        for account_node in component_accounts:

            account_id = account_node[1]

            cluster_sizes[account_id] = account_count

            account_degrees[account_id] = graph.degree(
                account_node
            )

            device_count = 0
            payment_count = 0
            ip_count = 0

            for identifier_node in graph.neighbors(
                account_node
            ):

                connected_accounts = [
                    neighbour
                    for neighbour in graph.neighbors(
                        identifier_node
                    )
                    if neighbour[0] == "account"
                ]

                # An identifier is shared only if it
                # connects more than one account.
                if len(connected_accounts) <= 1:
                    continue

                identifier_type = identifier_node[0]

                if identifier_type == "device":
                    device_count += 1

                elif identifier_type == "payment":
                    payment_count += 1

                elif identifier_type == "ip":
                    ip_count += 1

            shared_device_counts[account_id] = device_count
            shared_payment_counts[account_id] = payment_count
            shared_ip_counts[account_id] = ip_count

    # --------------------------------------------------
    # 3. ADD FEATURES TO DATAFRAME
    # --------------------------------------------------

    account_series = df["account_id"].astype("string")

    df["graph_cluster_size"] = (
        account_series.map(cluster_sizes)
        .fillna(1)
        .astype(int)
    )

    df["graph_node_degree"] = (
        account_series.map(account_degrees)
        .fillna(0)
        .astype(int)
    )

    df["shared_device_count"] = (
        account_series.map(shared_device_counts)
        .fillna(0)
        .astype(int)
    )

    df["shared_payment_count"] = (
        account_series.map(shared_payment_counts)
        .fillna(0)
        .astype(int)
    )

    df["shared_ip_count"] = (
        account_series.map(shared_ip_counts)
        .fillna(0)
        .astype(int)
    )

    df["shared_identifier_count"] = (
        df["shared_device_count"]
        + df["shared_payment_count"]
        + df["shared_ip_count"]
    )

    return df


# ======================================================
# TEST GRAPH ANALYSIS
# ======================================================

if __name__ == "__main__":

    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    train_path = (
        project_root / "data" / "train_accounts.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            "Training dataset not found. "
            "Run python -m src.generate_data first."
        )

    train_df = pd.read_csv(train_path)

    graph_df = build_account_graph(train_df)

    print("\n======================================")
    print("       TRIALGUARD GRAPH ANALYSIS")
    print("======================================")

    print("Total accounts:", len(graph_df))

    print(
        "Maximum account cluster:",
        graph_df["graph_cluster_size"].max()
    )

    print(
        "Accounts sharing devices:",
        (graph_df["shared_device_count"] > 0).sum()
    )

    print(
        "Accounts sharing payments:",
        (graph_df["shared_payment_count"] > 0).sum()
    )

    print(
        "Accounts sharing IP addresses:",
        (graph_df["shared_ip_count"] > 0).sum()
    )

    print("\nSample graph features:")

    print(
        graph_df[
            [
                "account_id",
                "graph_cluster_size",
                "shared_device_count",
                "shared_payment_count",
                "shared_ip_count",
                "shared_identifier_count"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nGraph analysis completed successfully!")
