"""Unit tests for CTraderTokenManager."""
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from live_trading.brokers.ctrader_token_manager import (
    CTraderTokenManager,
    REFRESH_MARGIN_SECONDS,
)


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    return tmp_path / ".ctrader_tokens.json"


@pytest.fixture
def manager(token_file: Path) -> CTraderTokenManager:
    return CTraderTokenManager(
        client_id="test_id",
        client_secret="test_secret",
        token_file=token_file,
    )


class TestTokenPersistence:
    def test_store_and_reload(self, token_file: Path):
        m = CTraderTokenManager(
            client_id="cid", client_secret="cs", token_file=token_file
        )
        m.store_tokens("at_123", "rt_456", expires_in=100)

        m2 = CTraderTokenManager(
            client_id="cid", client_secret="cs", token_file=token_file
        )
        assert m2._access_token == "at_123"
        assert m2._refresh_token == "rt_456"
        assert m2._expires_at > time.time()

    def test_loads_from_env_when_no_file(self, token_file: Path):
        with patch("live_trading.brokers.ctrader_token_manager.config") as mock_cfg:
            mock_cfg.CTRADER_CLIENT_ID = "cid"
            mock_cfg.CTRADER_CLIENT_SECRET = "cs"
            mock_cfg.CTRADER_ACCESS_TOKEN = "env_token"
            m = CTraderTokenManager(
                client_id="cid", client_secret="cs", token_file=token_file
            )
        assert m._access_token == "env_token"
        assert m._refresh_token is None
        assert m._expires_at == 0.0

    def test_file_takes_precedence_over_env(self, token_file: Path):
        token_file.write_text(json.dumps({
            "access_token": "file_tok",
            "refresh_token": "file_ref",
            "expires_at": time.time() + 99999,
        }))
        with patch("live_trading.brokers.ctrader_token_manager.config") as mock_cfg:
            mock_cfg.CTRADER_CLIENT_ID = "cid"
            mock_cfg.CTRADER_CLIENT_SECRET = "cs"
            mock_cfg.CTRADER_ACCESS_TOKEN = "env_token"
            m = CTraderTokenManager(
                client_id="cid", client_secret="cs", token_file=token_file
            )
        assert m._access_token == "file_tok"

    def test_corrupt_file_falls_back_to_env(self, token_file: Path):
        token_file.write_text("not json at all")
        with patch("live_trading.brokers.ctrader_token_manager.config") as mock_cfg:
            mock_cfg.CTRADER_CLIENT_ID = "cid"
            mock_cfg.CTRADER_CLIENT_SECRET = "cs"
            mock_cfg.CTRADER_ACCESS_TOKEN = "env_fallback"
            m = CTraderTokenManager(
                client_id="cid", client_secret="cs", token_file=token_file
            )
        assert m._access_token == "env_fallback"


class TestExpiry:
    def test_fresh_token_is_not_expired(self, manager: CTraderTokenManager):
        manager.store_tokens("at", "rt", expires_in=100_000)
        assert not manager._is_expired()

    def test_old_token_is_expired(self, manager: CTraderTokenManager):
        manager._access_token = "old"
        manager._expires_at = time.time() - 1
        assert manager._is_expired()

    def test_token_within_margin_is_expired(self, manager: CTraderTokenManager):
        manager._access_token = "soon"
        manager._expires_at = time.time() + REFRESH_MARGIN_SECONDS - 1
        assert manager._is_expired()

    def test_no_token_is_expired(self, manager: CTraderTokenManager):
        manager._access_token = None
        assert manager._is_expired()


class TestRefresh:
    @patch("live_trading.brokers.ctrader_token_manager.requests.get")
    def test_successful_refresh(self, mock_get, manager: CTraderTokenManager, token_file: Path):
        manager._refresh_token = "old_rt"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "accessToken": "new_at",
            "refreshToken": "new_rt",
            "expiresIn": 2_628_000,
            "errorCode": None,
        }
        mock_get.return_value = mock_resp

        result = manager.force_refresh()

        assert result is True
        assert manager._access_token == "new_at"
        assert manager._refresh_token == "new_rt"
        assert manager._expires_at > time.time()

        saved = json.loads(token_file.read_text())
        assert saved["access_token"] == "new_at"
        assert saved["refresh_token"] == "new_rt"

    @patch("live_trading.brokers.ctrader_token_manager.requests.get")
    def test_refresh_api_error(self, mock_get, manager: CTraderTokenManager):
        manager._refresh_token = "rt"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "accessToken": None,
            "refreshToken": None,
            "errorCode": "INVALID_TOKEN",
            "description": "bad token",
        }
        mock_get.return_value = mock_resp

        assert manager.force_refresh() is False
        assert manager._access_token is None

    @patch("live_trading.brokers.ctrader_token_manager.requests.get")
    def test_refresh_http_failure(self, mock_get, manager: CTraderTokenManager):
        import requests as req
        manager._refresh_token = "rt"
        mock_get.side_effect = req.ConnectionError("network error")

        assert manager.force_refresh() is False

    def test_refresh_without_refresh_token(self, manager: CTraderTokenManager):
        manager._refresh_token = None
        assert manager.force_refresh() is False


class TestAccessTokenProperty:
    @patch("live_trading.brokers.ctrader_token_manager.requests.get")
    def test_auto_refreshes_on_access(self, mock_get, manager: CTraderTokenManager):
        manager._access_token = "expired"
        manager._refresh_token = "rt"
        manager._expires_at = time.time() - 1

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "accessToken": "fresh_at",
            "refreshToken": "fresh_rt",
            "expiresIn": 2_628_000,
        }
        mock_get.return_value = mock_resp

        token = manager.access_token
        assert token == "fresh_at"
        mock_get.assert_called_once()

    def test_returns_valid_token_without_refresh(self, manager: CTraderTokenManager):
        manager._access_token = "good"
        manager._expires_at = time.time() + 999999

        with patch("live_trading.brokers.ctrader_token_manager.requests.get") as mock_get:
            token = manager.access_token
            assert token == "good"
            mock_get.assert_not_called()

    def test_returns_empty_when_no_refresh_token_and_expired(self, manager: CTraderTokenManager):
        manager._access_token = "old"
        manager._refresh_token = None
        manager._expires_at = time.time() - 1

        token = manager.access_token
        assert token == "old"


class TestHasRefreshToken:
    def test_true_when_present(self, manager: CTraderTokenManager):
        manager._refresh_token = "rt"
        assert manager.has_refresh_token() is True

    def test_false_when_missing(self, manager: CTraderTokenManager):
        manager._refresh_token = None
        assert manager.has_refresh_token() is False

    def test_false_when_empty_string(self, manager: CTraderTokenManager):
        manager._refresh_token = ""
        assert manager.has_refresh_token() is False
