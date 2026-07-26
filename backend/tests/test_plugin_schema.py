"""
Unit tests for backend/app/plugins/schema.py -- RegulationBody, RegulationFile, and helpers.

Copyright (C) 2024 Sarthak Doshi (github.com/SdSarthak)
SPDX-License-Identifier: AGPL-3.0-only
"""

import pytest

from app.plugins.schema import (
    ComplianceQuestion,
    RegulationBody,
    RegulationFile,
    RegulationBody,
    RiskFactor,
)


class TestRiskFactor:
    def test_valid_risk_factor(self):
        """A RiskFactor with all required fields is accepted."""
        rf = RiskFactor(
            id="rf1",
            label="High Risk",
            severity="high",
        )
        assert rf.id == "rf1"
        assert rf.label == "High Risk"
        assert rf.severity == "high"

    def test_extra_fields_rejected(self):
        """RiskFactor forbids extra fields via model_config extra='forbid'."""
        with pytest.raises(Exception):  # pydantic.ValidationError
            RiskFactor(
                id="rf1",
                label="High Risk",
                severity="high",
                unknown_field="reject",
            )


class TestRegulationBody:
    def _make_body(
        self,
        risk_factors=None,
        prohibited_uses=None,
        compliance_questions=None,
        required_documents=None,
    ):
        if risk_factors is None:
            risk_factors = [
                RiskFactor(id="rf1", label="High Risk", severity="high"),
            ]
        if prohibited_uses is None:
            prohibited_uses = ["Use A", "Use B"]
        if compliance_questions is None:
            compliance_questions = [
                ComplianceQuestion(id="q1", text="Is this safe?", maps_to="rf1"),
            ]
        if required_documents is None:
            required_documents = ["Risk Assessment Report"]
        return RegulationBody(
            name="EU AI Act",
            version="1.0",
            risk_factors=risk_factors,
            prohibited_uses=prohibited_uses,
            compliance_questions=compliance_questions,
            required_documents=required_documents,
        )

    def test_valid_body_passes(self):
        """A RegulationBody with all required fields passes validation."""
        body = self._make_body()
        assert body.name == "EU AI Act"
        assert body.version == "1.0"
        assert len(body.risk_factors) == 1

    def test_empty_risk_factors_rejected(self):
        """risk_factors list with min_length=1 rejects empty list."""
        with pytest.raises(Exception):  # pydantic.ValidationError
            RegulationBody(
                name="EU AI Act",
                version="1.0",
                risk_factors=[],
                prohibited_uses=["Use A"],
                compliance_questions=[
                    ComplianceQuestion(id="q1", text="?", maps_to="nonexistent")
                ],
                required_documents=["Doc 1"],
            )

    def test_empty_prohibited_uses_rejected(self):
        """prohibited_uses list with min_length=1 rejects empty list."""
        with pytest.raises(Exception):
            RegulationBody(
                name="EU AI Act",
                version="1.0",
                risk_factors=[RiskFactor(id="rf1", label="X", severity="high")],
                prohibited_uses=[],
                compliance_questions=[
                    ComplianceQuestion(id="q1", text="?", maps_to="rf1"),
                ],
                required_documents=["Doc 1"],
            )

    def test_empty_compliance_questions_rejected(self):
        """compliance_questions list with min_length=1 rejects empty list."""
        with pytest.raises(Exception):
            RegulationBody(
                name="EU AI Act",
                version="1.0",
                risk_factors=[RiskFactor(id="rf1", label="X", severity="high")],
                prohibited_uses=["Use A"],
                compliance_questions=[],
                required_documents=["Doc 1"],
            )

    def test_invalid_maps_to_raises(self):
        """A ComplianceQuestion with maps_to referencing non-existent risk_factor raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            RegulationBody(
                name="EU AI Act",
                version="1.0",
                risk_factors=[RiskFactor(id="rf1", label="X", severity="high")],
                prohibited_uses=["Use A"],
                compliance_questions=[
                    ComplianceQuestion(id="q1", text="?", maps_to="nonexistent_risk_factor"),
                ],
                required_documents=["Doc 1"],
            )
        assert "Invalid maps_to references" in str(exc_info.value)
        assert "nonexistent_risk_factor" in str(exc_info.value)

    def test_valid_maps_to_passes(self):
        """ComplianceQuestion with maps_to pointing to an existing risk_factor passes."""
        body = self._make_body(
            risk_factors=[RiskFactor(id="rf1", label="High", severity="high")],
            compliance_questions=[
                ComplianceQuestion(id="q1", text="Is it safe?", maps_to="rf1"),
                ComplianceQuestion(id="q2", text="Any concerns?", maps_to="rf1"),
            ],
        )
        assert len(body.compliance_questions) == 2

    def test_get_risk_factor_returns_existing(self):
        """get_risk_factor returns the matching RiskFactor."""
        body = self._make_body(
            risk_factors=[
                RiskFactor(id="rf1", label="High", severity="high"),
                RiskFactor(id="rf2", label="Low", severity="minimal"),
            ],
        )
        rf = body.get_risk_factor("rf1")
        assert rf is not None
        assert rf.id == "rf1"
        assert rf.label == "High"

    def test_get_risk_factor_returns_none_for_unknown(self):
        """get_risk_factor returns None for an id not in risk_factors."""
        body = self._make_body()
        assert body.get_risk_factor("nonexistent") is None

    def test_extra_fields_rejected(self):
        """RegulationBody forbids extra fields."""
        with pytest.raises(Exception):
            RegulationBody(
                name="EU AI Act",
                version="1.0",
                risk_factors=[RiskFactor(id="rf1", label="X", severity="high")],
                prohibited_uses=["Use A"],
                compliance_questions=[
                    ComplianceQuestion(id="q1", text="?", maps_to="rf1"),
                ],
                required_documents=["Doc 1"],
                unknown_extra_field="should reject",
            )


class TestRegulationFile:
    def test_wraps_regulation_body(self):
        """RegulationFile correctly wraps a RegulationBody."""
        body = RegulationBody(
            name="GDPR",
            version="2.0",
            risk_factors=[RiskFactor(id="rf1", label="High", severity="high")],
            prohibited_uses=["Tracking without consent"],
            compliance_questions=[
                ComplianceQuestion(id="q1", text="Consent?", maps_to="rf1"),
            ],
            required_documents=["DPIA"],
        )
        file = RegulationFile(regulation=body)
        assert file.regulation is body
        assert file.regulation.name == "GDPR"
        assert file.regulation.version == "2.0"

    def test_extra_fields_rejected(self):
        """RegulationFile forbids extra fields."""
        body = RegulationBody(
            name="GDPR",
            version="2.0",
            risk_factors=[RiskFactor(id="rf1", label="High", severity="high")],
            prohibited_uses=["Tracking without consent"],
            compliance_questions=[
                ComplianceQuestion(id="q1", text="Consent?", maps_to="rf1"),
            ],
            required_documents=["DPIA"],
        )
        with pytest.raises(Exception):
            RegulationFile(regulation=body, extra_field="reject")

    def test_invalid_body_raises(self):
        """RegulationFile rejects an invalid RegulationBody."""
        with pytest.raises(Exception):
            RegulationFile(
                regulation={
                    "name": "GDPR",
                    "version": "2.0",
                    "risk_factors": [],
                    "prohibited_uses": [],
                    "compliance_questions": [],
                    "required_documents": [],
                    # missing required fields -- will raise ValidationError
                }
            )
