
import networkx as nx
import pandas as pd


def build_account_graph(df):

    df = df.copy()

    required_columns = [
        "account_id",
        "device_hash",
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

    for _, row in df.iterrows():

        if pd.isna(row["account_id"]):
            continue

        account_id = str(row["account_id"]).strip()

        if not account_id:
            continue

        account_node = ("account", account_id)
        graph.add_node(account_node)

        identifiers = [
            ("device", row["device_hash"]),
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

            graph.add_edge(
                account_node,
                identifier_node
            )

    cluster_sizes = {}
    account_degrees = {}
    shared_devices = {}
    shared_ips = {}

    for component in nx.connected_components(graph):

        accounts = [
            node for node in component
            if node[0] == "account"
        ]

        for account_node in accounts:

            account_id = account_node[1]

            cluster_sizes[account_id] = len(accounts)

            account_degrees[account_id] = (
                graph.degree(account_node)
            )

            device_count = 0
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

                if len(connected_accounts) <= 1:
                    continue

                if identifier_node[0] == "device":
                    device_count += 1

                elif identifier_node[0] == "ip":
                    ip_count += 1

            shared_devices[account_id] = device_count
            shared_ips[account_id] = ip_count

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
        account_series.map(shared_devices)
        .fillna(0)
        .astype(int)
    )

    df["shared_ip_count"] = (
        account_series.map(shared_ips)
        .fillna(0)
        .astype(int)
    )

    # Temporary compatibility column for the old dashboard.
    df["shared_payment_count"] = 0

    df["shared_identifier_count"] = (
        df["shared_device_count"]
        + df["shared_ip_count"]
    )

    return df
