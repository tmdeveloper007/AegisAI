"""
Unit tests for backend/app/modules/guard/guard_config.py

Tests cover:
  - _has_model_weights correctly detects model weight files
  - get_trained_model_path falls back to CLASSIFIER_MODEL_PATH when no weights found
  - get_trained_model_path returns alternative path when weights found there
  - Constants (INTENT_CLASSES, thresholds) are correctly defined
"""

from __future__ import annotations

import os
import tempfile

import pytest


class TestHasModelWeights:
    """Tests for _has_model_weights function."""

    def test_returns_true_when_pytorch_model_bin_exists(self, tmp_path):
        """pytorch_model.bin presence should cause _has_model_weights to return True."""
        from app.modules.guard import guard_config as config_module

        model_dir = str(tmp_path)
        with open(os.path.join(model_dir, "pytorch_model.bin"), "w") as f:
            f.write("fake weights")

        result = config_module._has_model_weights(model_dir)
        assert result is True

    def test_returns_true_when_safetensors_exists(self, tmp_path):
        """model.safetensors presence should cause _has_model_weights to return True."""
        from app.modules.guard import guard_config as config_module

        model_dir = str(tmp_path)
        with open(os.path.join(model_dir, "model.safetensors"), "w") as f:
            f.write("fake safetensors")

        result = config_module._has_model_weights(model_dir)
        assert result is True

    def test_returns_false_when_neither_file_exists(self, tmp_path):
        """Neither pytorch_model.bin nor model.safetensors should return False."""
        from app.modules.guard import guard_config as config_module

        result = config_module._has_model_weights(str(tmp_path))
        assert result is False

    def test_returns_false_when_dir_does_not_exist(self):
        """Non-existent directory should return False."""
        from app.modules.guard import guard_config as config_module

        result = config_module._has_model_weights("/nonexistent/path/for/test")
        assert result is False


class TestGetTrainedModelPath:
    """Tests for get_trained_model_path function."""

    def test_returns_classifier_model_path_by_default(self):
        """When no weights found anywhere, should return the configured CLASSIFIER_MODEL_PATH."""
        from app.modules.guard import guard_config as config_module

        result = config_module.get_trained_model_path()
        assert result == config_module.CLASSIFIER_MODEL_PATH

    def test_returns_alt_path_when_weights_found(self, monkeypatch, tmp_path):
        """When alternative path has weights, should return that path."""
        from app.modules.guard import guard_config as config_module

        # Create a classifier directory with weights inside tmp_path
        classifier_dir = tmp_path / "classifier"
        classifier_dir.mkdir()
        (classifier_dir / "pytorch_model.bin").write_text("fake weights")

        # Override MODELS_DIR to point to our temp location
        monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path)

        # Make CLASSIFIER_MODEL_PATH point somewhere without weights
        monkeypatch.setattr(
            config_module,
            "CLASSIFIER_MODEL_PATH",
            str(tmp_path / "nowhere"),
        )

        result = config_module.get_trained_model_path()
        assert result == str(classifier_dir)


class TestConstants:
    """Tests for module-level constants."""

    def test_intent_classes_defined(self):
        """INTENT_CLASSES should contain the expected values."""
        from app.modules.guard import guard_config as config_module

        assert config_module.INTENT_CLASSES == ["benign", "suspicious", "malicious"]

    def test_intent_to_id_mapping(self):
        """INTENT_TO_ID should map intent names to integer IDs."""
        from app.modules.guard import guard_config as config_module

        assert config_module.INTENT_TO_ID == {
            "benign": 0,
            "suspicious": 1,
            "malicious": 2,
        }

    def test_id_to_intent_mapping(self):
        """ID_TO_INTENT should be the reverse of INTENT_TO_ID."""
        from app.modules.guard import guard_config as config_module

        assert config_module.ID_TO_INTENT == {v: k for k, v in config_module.INTENT_TO_ID.items()}

    def test_thresholds_are_positive_floats(self):
        """Threshold constants should be positive float values."""
        from app.modules.guard import guard_config as config_module

        assert isinstance(config_module.INTENT_CLASSIFIER_THRESHOLD, float)
        assert config_module.INTENT_CLASSIFIER_THRESHOLD > 0
        assert isinstance(config_module.SUSPICIOUS_THRESHOLD, float)
        assert config_module.SUSPICIOUS_THRESHOLD > 0
        assert isinstance(config_module.MALICIOUS_THRESHOLD, float)
        assert config_module.MALICIOUS_THRESHOLD > 0
        assert config_module.MALICIOUS_THRESHOLD > config_module.SUSPICIOUS_THRESHOLD

    def test_max_prompt_length_is_positive_int(self):
        """MAX_PROMPT_LENGTH should be a positive integer."""
        from app.modules.guard import guard_config as config_module

        assert isinstance(config_module.MAX_PROMPT_LENGTH, int)
        assert config_module.MAX_PROMPT_LENGTH > 0

    def test_sanitization_level_valid(self):
        """SANITIZATION_LEVEL should be one of the allowed values."""
        from app.modules.guard import guard_config as config_module

        assert config_module.SANITIZATION_LEVEL in ("low", "medium", "high")
