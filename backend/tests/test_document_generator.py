"""
Unit tests for backend/app/modules/llm/document_generator.py -- generate_compliance_narrative.

Copyright (C) 2024 Sarthak Doshi (github.com/SdSarthak)
SPDX-License-Identifier: AGPL-3.0-only
"""

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/test")
os.environ.setdefault("SECRET_KEY", "testsecret")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("DEBUG", "True")

from app.models.ai_system import AISystem, RiskLevel, RiskAssessment
from app.models.document import DocumentType
from app.modules.llm.document_generator import generate_compliance_narrative


class _DummyDoc:
    def __init__(self, content: str):
        self.page_content = content


def _make_ai_system(**overrides):
    defaults = {
        "id": 1,
        "name": "Test AI System",
        "version": "1.0",
        "use_case": "Automated hiring",
        "sector": "Human Resources",
        "risk_level": RiskLevel.HIGH,
        "description": "A system for screening job candidates.",
        "owner_id": 1,
    }
    defaults.update(overrides)
    mock_obj = MagicMock(spec=AISystem)
    for k, v in defaults.items():
        setattr(mock_obj, k, v)
    return mock_obj


class TestGenerateComplianceNarrative:
    def test_filenotfound_error_returns_fallback_context(self):
        """FileNotFoundError from load_vector_store is caught and a fallback message is used."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.side_effect = FileNotFoundError("No index found")

        with patch(
            "app.modules.llm.document_generator.load_vector_store",
            return_value=mock_vs,
        ):
            with patch(
                "app.modules.llm.document_generator.LLMClient"
            ) as mock_client_cls:
                mock_client = MagicMock()
                mock_client.call.return_value = "Generated document text."
                mock_client_cls.return_value = mock_client

                result = generate_compliance_narrative(
                    document_type=DocumentType.RISK_ASSESSMENT,
                    ai_system=_make_ai_system(),
                    risk_assessment=None,
                    company_name="TestCorp",
                    user_id=1,
                )

        assert "Generated document text." in result
        # Verify LLM was called (fallback was used, not from similarity_search)
        mock_client.call.assert_called_once()

    def test_normal_execution_with_mocked_rag_and_llm(self):
        """When RAG returns docs and LLM succeeds, a narrative string is returned."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            _DummyDoc("Article 5 of EU AI Act prohibits certain AI practices."),
            _DummyDoc("High-risk AI systems must comply with Article 9 requirements."),
        ]

        with patch(
            "app.modules.llm.document_generator.load_vector_store",
            return_value=mock_vs,
        ):
            with patch(
                "app.modules.llm.document_generator.LLMClient"
            ) as mock_client_cls:
                mock_client = MagicMock()
                mock_client.call.return_value = "Professional compliance narrative text."
                mock_client_cls.return_value = mock_client

                result = generate_compliance_narrative(
                    document_type=DocumentType.RISK_ASSESSMENT,
                    ai_system=_make_ai_system(),
                    risk_assessment=None,
                    company_name="Acme Corp",
                    user_id=1,
                )

        assert result == "Professional compliance narrative text."
        # Check RAG was called
        mock_vs.similarity_search.assert_called_once()

    def test_company_name_included_in_prompt(self):
        """The company_name appears in the prompt sent to the LLM."""
        captured_prompt = {}

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        with patch(
            "app.modules.llm.document_generator.load_vector_store",
            return_value=mock_vs,
        ):
            with patch(
                "app.modules.llm.document_generator.LLMClient"
            ) as mock_client_cls:
                mock_client = MagicMock()

                def capture_call(prompt, **kwargs):
                    captured_prompt["prompt"] = prompt
                    return "output"

                mock_client.call.side_effect = capture_call
                mock_client_cls.return_value = mock_client

                generate_compliance_narrative(
                    document_type=DocumentType.TECHNICAL_DOCUMENTATION,
                    ai_system=_make_ai_system(),
                    risk_assessment=None,
                    company_name="GlobalTech Ltd",
                    user_id=None,
                )

        assert "GlobalTech Ltd" in captured_prompt.get("prompt", "")

    def test_system_name_included_in_prompt(self):
        """The AI system name appears in the prompt sent to the LLM."""
        captured_prompt = {}

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        with patch(
            "app.modules.llm.document_generator.load_vector_store",
            return_value=mock_vs,
        ):
            with patch(
                "app.modules.llm.document_generator.LLMClient"
            ) as mock_client_cls:
                mock_client = MagicMock()

                def capture_call(prompt, **kwargs):
                    captured_prompt["prompt"] = prompt
                    return "output"

                mock_client.call.side_effect = capture_call
                mock_client_cls.return_value = mock_client

                generate_compliance_narrative(
                    document_type=DocumentType.RISK_ASSESSMENT,
                    ai_system=_make_ai_system(name="HireAssist AI"),
                    risk_assessment=None,
                    company_name="Acme",
                    user_id=None,
                )

        assert "HireAssist AI" in captured_prompt.get("prompt", "")

    def test_document_type_included_in_prompt(self):
        """The document_type appears in the prompt sent to the LLM."""
        captured_prompt = {}

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        with patch(
            "app.modules.llm.document_generator.load_vector_store",
            return_value=mock_vs,
        ):
            with patch(
                "app.modules.llm.document_generator.LLMClient"
            ) as mock_client_cls:
                mock_client = MagicMock()

                def capture_call(prompt, **kwargs):
                    captured_prompt["prompt"] = prompt
                    return "output"

                mock_client.call.side_effect = capture_call
                mock_client_cls.return_value = mock_client

                generate_compliance_narrative(
                    document_type=DocumentType.RISK_ASSESSMENT,
                    ai_system=_make_ai_system(),
                    risk_assessment=None,
                    company_name="Acme",
                    user_id=None,
                )

        # Risk Assessment is the document type value
        assert "risk_assessment" in captured_prompt.get("prompt", "").lower()
