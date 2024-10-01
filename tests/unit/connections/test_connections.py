from unittest.mock import MagicMock, patch

import pytest

from fiboaitech.connections.connections import Qdrant as QdrantConnection


@pytest.fixture
def mock_qdrant_env_vars(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://mocked_qdrant_url")
    monkeypatch.setenv("QDRANT_API_KEY", "mocked_api_key")


def test_qdrant_initialization_with_env_vars(mock_qdrant_env_vars):
    qdrant = QdrantConnection()
    assert qdrant.url == "http://mocked_qdrant_url"
    assert qdrant.api_key == "mocked_api_key"


def test_qdrant_initialization_with_provided_values():
    qdrant = QdrantConnection(url="http://custom_qdrant_url", api_key="custom_api_key")
    assert qdrant.url == "http://custom_qdrant_url"
    assert qdrant.api_key == "custom_api_key"


@patch("qdrant_client.QdrantClient")
def test_qdrant_connect(mock_qdrant_client_class, mock_qdrant_env_vars):
    mock_qdrant_client_instance = MagicMock()
    mock_qdrant_client_class.return_value = mock_qdrant_client_instance

    qdrant = QdrantConnection()
    client = qdrant.connect()

    mock_qdrant_client_class.assert_called_once_with(url="http://mocked_qdrant_url", api_key="mocked_api_key")
    assert client == mock_qdrant_client_instance
