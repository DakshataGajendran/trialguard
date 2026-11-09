import pandas as pd

from src.graph import build_account_graph


def test_graph_detects_shared_device():
    data = pd.DataFrame({
        "account_id": ["A001", "A002", "A003"],
        "device_hash": ["DEVICE_X", "DEVICE_X", "DEVICE_Y"],
        "payment_hash": ["PAY_1", "PAY_2", "PAY_3"],
        "ip_address": ["192.168.1.1", "192.168.1.2", "192.168.1.3"]
    })

    result = build_account_graph(data)

    assert "shared_device_count" in result.columns

    account_one = result[result["account_id"] == "A001"].iloc[0]
    account_two = result[result["account_id"] == "A002"].iloc[0]

    assert account_one["shared_device_count"] > 0
    assert account_two["shared_device_count"] > 0


def test_graph_does_not_use_payment_for_detection():
    data = pd.DataFrame({
        "account_id": ["A001", "A002"],
        "device_hash": ["DEVICE_1", "DEVICE_2"],
        "payment_hash": ["PAY_SHARED", "PAY_SHARED"],
        "ip_address": ["192.168.1.1", "192.168.1.2"]
    })

    result = build_account_graph(data)

    assert "shared_payment_count" in result.columns
    assert result["shared_payment_count"].sum() == 0


def test_graph_returns_cluster_information():
    data = pd.DataFrame({
        "account_id": ["A001", "A002"],
        "device_hash": ["DEVICE_X", "DEVICE_X"],
        "payment_hash": ["PAY_1", "PAY_2"],
        "ip_address": ["192.168.1.1", "192.168.1.2"]
    })

    result = build_account_graph(data)

    required_columns = [
        "graph_cluster_size",
        "graph_node_degree",
        "shared_device_count",
        "shared_payment_count",
        "shared_ip_count",
        "shared_identifier_count"
    ]

    for column in required_columns:
        assert column in result.columns

    assert len(result) == 2