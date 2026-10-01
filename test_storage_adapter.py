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


def test_latest_exchange_key_round_trip(storage, monkeypatch):
    objects = {}
    monkeypatch.setattr(storage, "_blob_put", lambda path, data, content_type="application/json": objects.__setitem__(path, data) or True)
    monkeypatch.setattr(storage, "_blob_get", lambda path: objects.get(path))
    monkeypatch.setattr(storage, "_encrypt", lambda value: "enc:" + value if value else "")
    assert storage.remote_save_latest_exchange_key(7, "mexc", "key7", "secret7", "", testnet=False)
    record = storage.remote_get_latest_exchange_key()
    assert record["exchange_id"] == "mexc"
    assert record["api_key"] == "enc:key7"
    assert record["testnet"] is False


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


def test_remote_delete_exchange_key(storage, monkeypatch):
    objects = {}
    monkeypatch.setattr(storage, "_blob_put", lambda path, data, content_type="application/json": objects.__setitem__(path, data) or True)
    monkeypatch.setattr(storage, "_blob_get", lambda path: objects.get(path))
    monkeypatch.setattr(storage, "_blob_del", lambda path: objects.pop(path, None) is not None)
    monkeypatch.setattr(storage, "_encrypt", lambda value: value)

    assert storage.remote_save_exchange_key(1, "binance", "key", "secret")
    assert storage.remote_save_latest_exchange_key(1, "binance", "key", "secret", "")
    assert "exchange_keys/1/binance.json" in objects
    assert "exchange_keys/latest.json" in objects

    assert storage.remote_delete_exchange_key(1, "binance") is True
    assert "exchange_keys/1/binance.json" not in objects
    assert "exchange_keys/latest.json" not in objects


def test_remote_app_setting_round_trip(storage, monkeypatch):
    objects = {}
    monkeypatch.setattr(storage, "_blob_put", lambda path, data, content_type="application/json": objects.__setitem__(path, data) or True)
    monkeypatch.setattr(storage, "_blob_get", lambda path: objects.get(path))

    assert storage.remote_save_app_setting("execution_mode", "manual")
    assert storage.remote_get_app_setting("execution_mode") == "manual"


def test_storage_adapter_supports_none_and_disabled_drivers(monkeypatch):
    import storage_adapter
    for driver_name in ("none", "off", "disabled", "local", "sqlite"):
        monkeypatch.setenv("REMOTE_STORAGE_DRIVER", driver_name)
        reloaded = importlib.reload(storage_adapter)
        assert reloaded._driver() == "none"
        assert reloaded.is_remote_storage_available() is False


def test_storage_adapter_suspends_blob_on_http_403(storage, monkeypatch):
    import urllib.error
    monkeypatch.setattr(storage, "_BLOB_SUSPENDED", False)

    def raise_403(*args, **kwargs):
        raise urllib.error.HTTPError("http://example", 403, "Forbidden", {}, None)

    monkeypatch.setattr(storage.urllib.request, "urlopen", raise_403)
    res = storage._blob_get("users/index.json")
    assert res is None
    assert storage._BLOB_SUSPENDED is True
    assert storage.is_remote_storage_available() is False


def test_remote_app_setting_ttl_caching(storage, monkeypatch):
    call_count = {"get": 0}
    storage._APP_SETTINGS_CACHE.clear()

    def mock_blob_get(path):
        call_count["get"] += 1
        return b'"auto"'

    monkeypatch.setattr(storage, "_blob_get", mock_blob_get)
    val1 = storage.remote_get_app_setting("execution_mode")
    assert val1 == "auto"
    assert call_count["get"] == 1

    # Second call within TTL must hit cache without invoking mock_blob_get
    val2 = storage.remote_get_app_setting("execution_mode")
    assert val2 == "auto"
    assert call_count["get"] == 1

