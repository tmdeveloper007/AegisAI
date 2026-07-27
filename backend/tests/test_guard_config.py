"""Regression tests for guard_config path resolution.

These tests prevent regressions in path variables that previously broke —
DATA_DIR, MODELS_DIR, BACKEND_ROOT, and get_trained_model_path().
"""

from pathlib import Path
from unittest.mock import patch

import importlib.util, sys

spec = importlib.util.spec_from_file_location(
    "guard_config",
    __import__("pathlib").Path(__file__).parent.parent
    / "app/modules/guard/guard_config.py",
)
guard_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_config)


_GUARD_CONFIG_FILE = Path(guard_config.__file__).resolve()
_EXPECTED_BACKEND_ROOT = _GUARD_CONFIG_FILE.parent.parent.parent.parent


class TestPathResolution:
    def test_backend_root_resolves_to_backend(self):
        assert guard_config.BACKEND_ROOT.resolve() == _EXPECTED_BACKEND_ROOT

    def test_data_dir_resolves_under_backend(self):
        assert guard_config.DATA_DIR.resolve() == _EXPECTED_BACKEND_ROOT / "data"

    def test_models_dir_resolves_under_guard(self):
        expected = _GUARD_CONFIG_FILE.parent / "models"
        assert guard_config.MODELS_DIR.resolve() == expected.resolve()

    def test_backend_root_name_is_backend(self):
        assert guard_config.BACKEND_ROOT.name in ("backend", "app")

    def test_data_dir_name_is_data(self):
        assert guard_config.DATA_DIR.name == "data"

    def test_models_dir_name_is_models(self):
        assert guard_config.MODELS_DIR.name == "models"


class TestGetTrainedModelPath:
    """Tests for get_trained_model_path() fallback logic."""

    def _mock_exists(self, path: str) -> bool:
        """Return True only for CLASSIFIER_MODEL_PATH."""
        return path == guard_config.CLASSIFIER_MODEL_PATH

    def _mock_alt_exists(self, path: str) -> bool:
        """Return True for MODELS_DIR/classifier."""
        expected = str(guard_config.MODELS_DIR / "classifier")
        return path == expected

    def test_returns_env_path_when_classifier_model_path_exists(self):
        """When CLASSIFIER_MODEL_PATH exists, that path is returned."""
        with patch("os.path.exists", return_value=True):
            result = guard_config.get_trained_model_path()
        assert result == guard_config.CLASSIFIER_MODEL_PATH

    def test_falls_back_to_models_classifier_when_primary_missing(self):
        """When CLASSIFIER_MODEL_PATH does not exist but MODELS_DIR/classifier has weights."""
        # os.path.exists returns False for CLASSIFIER_MODEL_PATH,
        # True only for MODELS_DIR/classifier
        def mock_exists(path: str) -> bool:
            alt_classifier = str(guard_config.MODELS_DIR / "classifier")
            return path == alt_classifier

        def mock_has_weights(path: str) -> bool:
            return path == str(guard_config.MODELS_DIR / "classifier")

        with patch("os.path.exists", side_effect=mock_exists):
            with patch.object(guard_config, "_has_model_weights", side_effect=mock_has_weights):
                result = guard_config.get_trained_model_path()
        assert result == str(guard_config.MODELS_DIR / "classifier")

    def test_falls_back_to_intent_classifier_when_primary_and_alt_missing(self):
        """When neither CLASSIFIER_MODEL_PATH nor MODELS_DIR/classifier have weights."""
        def mock_exists(path: str) -> bool:
            return path == "./intent_classifier"

        def mock_has_weights(path: str) -> bool:
            return path == "./intent_classifier"

        with patch("os.path.exists", side_effect=mock_exists):
            with patch.object(guard_config, "_has_model_weights", side_effect=mock_has_weights):
                result = guard_config.get_trained_model_path()
        assert result == "./intent_classifier"

    def test_returns_classifier_model_path_when_no_path_has_weights(self):
        """When no path has model weights, returns the default CLASSIFIER_MODEL_PATH."""
        with patch("os.path.exists", return_value=False):
            result = guard_config.get_trained_model_path()
        assert result == guard_config.CLASSIFIER_MODEL_PATH

    def test_result_is_string(self):
        """Return value must be a non-empty string."""
        with patch("os.path.exists", return_value=False):
            result = guard_config.get_trained_model_path()
        assert isinstance(result, str)
        assert len(result) > 0
