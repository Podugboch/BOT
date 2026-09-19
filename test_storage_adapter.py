"""Security regression tests for the remote storage adapter."""
import importlib

import pytest


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.setenv("REMOTE_STORAGE_DRIVER", "blob")
    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "token")
    monkeypatch.setenv("BLOB_STORE_ID", "store_test-store")
    import storage_adapter
    return importlib.reload(storage_adapter)


def test_rejects_invalid_identifiers(storage):
    with pytest.raises(ValueError):
        storage._component("../account", "username")
    with pytest.raises(ValueError):
        storage._exchange("not-an-exchange")


def test_requires_explicit_blob_store(monkeypatch):
    monkeypatch.setenv("REMOTE_STORAGE_DRIVER", "blob")
    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "token")
    monkeypatch.delenv("BLOB_STORE_ID", raising=False)
    import storage_adapter
    with pytest.raises(RuntimeError):
        importlib.reload(storage_adapter)


def test_requires_https(monkeypatch):
    monkeypatch.setenv("KV_REST_API_URL", "http://redis.example")
    monkeypatch.setenv("KV_REST_API_TOKEN", "token")
    import storage_adapter
    with pytest.raises(RuntimeError):
        importlib.reload(storage_adapter)


def test_blob_exchange_index_and_masking(storage, monkeypatch):
    objects = {}
    monkeypatch.setattr(storage, "_blob_put", lambda path, data, content_type="application/json": objects.__setitem__(path, data) or True)
    monkeypatch.setattr(storage, "_blob_get", lambda path: objects.get(path))
    monkeypatch.setattr(storage, "_blob_del", lambda path: objects.pop(path, None) is not None)
    monkeypatch.setattr(storage, "_encrypt", lambda value: "enc:" + value if value else "")
    assert storage.remote_save_exchange_key(1, "binance", "key", "secret")
    records = storage.remote_get_exchange_keys(1)
    assert records[0]["api_secret"] == "enc:secret"
    assert storage.remote_get_exchange_keys(1, masked=True)[0]["api_secret"] == "••••"
    assert b"binance" in objects["exchange_keys/1/list.json"]


def test_failed_blob_write_is_reported(storage, monkeypatch):
    monkeypatch.setattr(storage, "_blob_put", lambda *args, **kwargs: False)
    assert storage.remote_save_user(1, "alice", "alice@example.com", "hash", "salt") is False


def test_alias_cleanup_and_account_delete(storage, monkeypatch):
    objects = {}
    monkeypatch.setattr(storage, "_blob_put", lambda path, data, content_type="application/json": objects.__setitem__(path, data) or True)
    monkeypatch.setattr(storage, "_blob_get", lambda path: objects.get(path))
    monkeypatch.setattr(storage, "_blob_del", lambda path: objects.pop(path, None) is not None)
    assert storage.remote_save_user(1, "oldname", "old@example.com", "hash", "salt")
    assert storage.remote_save_user(1, "newname", "new@example.com", "hash", "salt", previous_username="oldname", previous_email="old@example.com")
    assert "users/by_username/oldname.json" not in objects
    assert storage.remote_delete_account(1, "newname", "new@example.com")


def test_blob_concurrent_style_updates_preserve_index(storage, monkeypatch):
    objects = {}
    monkeypatch.setattr(storage, "_blob_put", lambda path, data, content_type="application/json": objects.__setitem__(path, data) or True)
    monkeypatch.setattr(storage, "_blob_get", lambda path: objects.get(path))
    assert storage.remote_save_user(1, "alice", "a@example.com", "h", "s")
    assert storage.remote_save_user(2, "bob", "b@example.com", "h", "s")
    index = storage._blob_json("users/index.json")
    assert set(index["ids"]) == {1, 2}
