"""
Unit tests for backend/app/plugins/regulation_loader.py

Tests cover:
  - RegulationRegistry._load_file parses valid YAML correctly
  - RegulationRegistry._load_file raises ValueError for invalid YAML
  - RegulationRegistry.get_regulation returns correct regulation
  - RegulationRegistry.get_regulation raises KeyError with available list
  - RegulationRegistry.list_regulations returns sorted list
  - Custom directory overrides built-in regulations
  - Module-level init_registry sets the global registry
  - Module-level get_registry raises RuntimeError if not initialized
  - Module-level get_regulation and list_regulations delegate correctly
"""

from __future__ import annotations

import pytest
import yaml

from app.plugins.regulation_loader import (
    RegulationRegistry,
    get_regulation,
    get_registry,
    init_registry,
    list_regulations,
)


# Minimal valid regulation YAML content
VALID_REGULATION_YAML = {
    "regulation": {
        "name": "EU AI Act",
        "version": "1.0",
        "risk_factors": [
            {"id": "rf1", "label": "Harm to Persons", "severity": "high"},
            {"id": "rf2", "label": "Property Damage", "severity": "limited"},
        ],
        "prohibited_uses": ["Social scoring by governments"],
        "required_documents": ["Technical documentation"],
        "compliance_questions": [
            {"id": "q1", "text": "Is the system high-risk?", "maps_to": "rf1"},
            {"id": "q2", "text": "Any prohibited use cases?", "maps_to": "rf2"},
        ],
    }
}

VALID_REGULATION_YAML_STR = yaml.safe_dump(VALID_REGULATION_YAML)


class TestRegulationRegistryLoadFile:
    """Tests for RegulationRegistry._load_file method."""

    def test_load_file_parses_valid_yaml(self, tmp_path):
        """_load_file should parse a valid regulation YAML and return RegulationBody."""
        reg_file = tmp_path / "eu_ai_act.yaml"
        reg_file.write_text(VALID_REGULATION_YAML_STR)

        # Create registry with empty builtin_dir to avoid loading during init
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        registry = RegulationRegistry(builtin_dir=empty_dir, custom_dir=None)
        regulation = registry._load_file(reg_file)

        assert regulation.name == "EU AI Act"
        assert regulation.version == "1.0"
        assert len(regulation.risk_factors) == 2
        assert regulation.risk_factors[0].id == "rf1"

    def test_load_file_propagates_yaml_error_for_malformed_yaml(self, tmp_path):
        """_load_file should propagate YAML parsing errors for malformed YAML."""
        reg_file = tmp_path / "invalid.yaml"
        reg_file.write_text("invalid: [unclosed")

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        registry = RegulationRegistry(builtin_dir=empty_dir, custom_dir=None)

        # yaml.YAMLError is raised for malformed YAML
        with pytest.raises(yaml.YAMLError):
            registry._load_file(reg_file)

    def test_load_file_raises_value_error_for_invalid_schema(self, tmp_path):
        """_load_file should raise ValueError for regulation missing required fields."""
        invalid_yaml = yaml.safe_dump({
            "regulation": {
                "name": "Test",
                # missing version, risk_factors, prohibited_uses, etc.
            }
        })
        reg_file = tmp_path / "invalid_schema.yaml"
        reg_file.write_text(invalid_yaml)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        registry = RegulationRegistry(builtin_dir=empty_dir, custom_dir=None)

        with pytest.raises(ValueError) as exc_info:
            registry._load_file(reg_file)
        assert "invalid_schema.yaml" in str(exc_info.value)


class TestRegulationRegistryLoadDir:
    """Tests for RegulationRegistry._load_dir method."""

    def test_load_dir_loads_all_yaml_files(self, tmp_path):
        """_load_dir should load every .yaml file in the directory."""
        files = {}
        for name in ["gdpr.yaml", "eu_ai_act.yaml"]:
            data = dict(VALID_REGULATION_YAML)
            data["regulation"]["name"] = name.replace(".yaml", "").upper()
            files[name] = yaml.safe_dump(data)

        for name, content in files.items():
            (tmp_path / name).write_text(content)

        registry = RegulationRegistry(builtin_dir=tmp_path, custom_dir=None)
        regs = registry.list_regulations()

        assert len(regs) == 2
        assert all(name.replace(".yaml", "").upper() in regs for name in files)


class TestRegulationRegistryGetRegulation:
    """Tests for RegulationRegistry.get_regulation method."""

    def test_get_regulation_returns_correct_body(self, tmp_path):
        """get_regulation should return the RegulationBody for a registered name."""
        reg_file = tmp_path / "gdpr.yaml"
        gdpr_data = dict(VALID_REGULATION_YAML)
        gdpr_data["regulation"]["name"] = "GDPR"
        reg_file.write_text(yaml.safe_dump(gdpr_data))

        registry = RegulationRegistry(builtin_dir=tmp_path, custom_dir=None)
        regulation = registry.get_regulation("GDPR")

        assert regulation.name == "GDPR"
        assert regulation.version == "1.0"

    def test_get_regulation_raises_keyerror_with_available_list(self, tmp_path):
        """get_regulation should raise KeyError listing available regulations."""
        reg_file = tmp_path / "gdpr.yaml"
        gdpr_data = dict(VALID_REGULATION_YAML)
        gdpr_data["regulation"]["name"] = "GDPR"
        reg_file.write_text(yaml.safe_dump(gdpr_data))

        registry = RegulationRegistry(builtin_dir=tmp_path, custom_dir=None)

        with pytest.raises(KeyError) as exc_info:
            registry.get_regulation("NonExistent")
        error_msg = str(exc_info.value)
        assert "NonExistent" in error_msg
        assert "GDPR" in error_msg


class TestRegulationRegistryListRegulations:
    """Tests for RegulationRegistry.list_regulations method."""

    def test_list_regulations_returns_sorted_list(self, tmp_path):
        """list_regulations should return regulation names in sorted order."""
        for name, label in [("zebra.yaml", "Zebra"), ("apple.yaml", "Apple")]:
            data = dict(VALID_REGULATION_YAML)
            data["regulation"]["name"] = label
            (tmp_path / name).write_text(yaml.safe_dump(data))

        registry = RegulationRegistry(builtin_dir=tmp_path, custom_dir=None)
        regs = registry.list_regulations()

        assert regs == sorted(regs)

    def test_list_regulations_returns_empty_for_empty_dir(self, tmp_path):
        """list_regulations should return empty list when no regulations loaded."""
        registry = RegulationRegistry(builtin_dir=tmp_path, custom_dir=None)
        assert registry.list_regulations() == []


class TestRegulationRegistryCustomOverrides:
    """Tests for custom directory override behavior."""

    def test_custom_overrides_builtin(self, tmp_path):
        """When custom dir has same-name regulation, it should override builtin."""
        builtin_file = tmp_path / "override_test.yaml"
        builtin_data = dict(VALID_REGULATION_YAML)
        builtin_data["regulation"]["name"] = "OverrideTest"
        builtin_data["regulation"]["version"] = "builtin"
        builtin_file.write_text(yaml.safe_dump(builtin_data))

        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        custom_file = custom_dir / "override_test.yaml"
        custom_data = dict(VALID_REGULATION_YAML)
        custom_data["regulation"]["name"] = "OverrideTest"
        custom_data["regulation"]["version"] = "custom"
        custom_file.write_text(yaml.safe_dump(custom_data))

        registry = RegulationRegistry(builtin_dir=tmp_path, custom_dir=custom_dir)
        regulation = registry.get_regulation("OverrideTest")

        assert regulation.version == "custom"


class TestModuleLevelFunctions:
    """Tests for module-level singleton functions."""

    def test_get_registry_raises_when_not_initialized(self):
        """get_registry should raise RuntimeError if init_registry was never called."""
        # Reset the global registry to None for this test
        import app.plugins.regulation_loader as loader_module
        original_registry = loader_module._registry
        loader_module._registry = None
        try:
            with pytest.raises(RuntimeError) as exc_info:
                get_registry()
            assert "init_registry()" in str(exc_info.value)
        finally:
            loader_module._registry = original_registry

    def test_init_registry_sets_global_registry(self, tmp_path):
        """init_registry should set the module-level _registry singleton."""
        reg_file = tmp_path / "test_reg.yaml"
        test_data = dict(VALID_REGULATION_YAML)
        test_data["regulation"]["name"] = "TestReg"
        reg_file.write_text(yaml.safe_dump(test_data))

        import app.plugins.regulation_loader as loader_module
        original_registry = loader_module._registry
        try:
            loader_module._registry = None
            registry = init_registry(builtin_dir=tmp_path, custom_dir=None)
            assert loader_module._registry is registry
            assert registry.get_regulation("TestReg").name == "TestReg"
        finally:
            loader_module._registry = original_registry

    def test_get_regulation_delegates_to_registry(self, tmp_path):
        """get_regulation should delegate to the global registry."""
        reg_file = tmp_path / "delegation_test.yaml"
        test_data = dict(VALID_REGULATION_YAML)
        test_data["regulation"]["name"] = "DelegationTest"
        reg_file.write_text(yaml.safe_dump(test_data))

        import app.plugins.regulation_loader as loader_module
        original_registry = loader_module._registry
        try:
            loader_module._registry = None
            init_registry(builtin_dir=tmp_path, custom_dir=None)
            result = get_regulation("DelegationTest")
            assert result.name == "DelegationTest"
        finally:
            loader_module._registry = original_registry

    def test_list_regulations_delegates_to_registry(self, tmp_path):
        """list_regulations should delegate to the global registry."""
        reg_file = tmp_path / "list_test.yaml"
        test_data = dict(VALID_REGULATION_YAML)
        test_data["regulation"]["name"] = "ListTest"
        reg_file.write_text(yaml.safe_dump(test_data))

        import app.plugins.regulation_loader as loader_module
        original_registry = loader_module._registry
        try:
            loader_module._registry = None
            init_registry(builtin_dir=tmp_path, custom_dir=None)
            result = list_regulations()
            assert "ListTest" in result
        finally:
            loader_module._registry = original_registry
