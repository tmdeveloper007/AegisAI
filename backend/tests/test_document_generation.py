"""
Unit tests for backend/app/modules/llm/document_generator.py
generate_compliance_narrative function.

Copyright (C) 2024 Sarthak Doshi (github.com/SdSarthak)
SPDX-License-Identifier: AGPL-3.0-only
"""

import pytest
from unittest.mock import patch, MagicMock

from app.modules.llm.document_generator import generate_compliance_narrative
from app.models.document import DocumentType
from app.models.ai_system import AISystem, RiskLevel


def _make_fake_ai_system(name="Test AI System", use_case="Customer Support", sector="Finance",
                          risk_level=RiskLevel.HIGH, description="A test AI system.",
                          version="1.0"):
    """Create a fake AISystem for testing."""
    system = MagicMock(spec=AISystem)
    system.name = name
    system.use_case = use_case
    system.sector = sector
    system.risk_level = risk_level
    system.description = description
    system.version = version
    return system


class TestGenerateComplianceNarrative:
    """Unit tests for generate_compliance_narrative."""

    def _mock_llm_client(self, return_text: str = "Generated compliance narrative."):
        """Return a mocked LLMClient that returns return_text."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = return_text

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_llm_client_instance = MagicMock()
        mock_llm_client_instance.call.return_value = return_text
        return mock_llm_client_instance

    @patch("app.modules.llm.document_generator.LLMClient")
    @patch("app.modules.llm.document_generator.load_vector_store")
    def test_file_not_found_error_fallback(self, mock_load_vs, mock_llm_cls):
        """FileNotFoundError from vector store should fall back gracefully."""
        mock_load_vs.side_effect = FileNotFoundError("Vector store not initialized")
        mock_llm_cls.return_value = self._mock_llm_client("Fallback narrative")

        system = _make_fake_ai_system()
        result = generate_compliance_narrative(
            document_type=DocumentType.TECHNICAL_DOCUMENTATION,
            ai_system=system,
            risk_assessment=None,
            company_name="Test Corp",
            user_id=None,
        )

        assert isinstance(result, str)
        assert result == "Fallback narrative"
        # Verify load_vector_store was called
        mock_load_vs.assert_called_once_with(user_id=None)

    @patch("app.modules.llm.document_generator.LLMClient")
    @patch("app.modules.llm.document_generator.load_vector_store")
    def test_normal_call_with_rag_context(self, mock_load_vs, mock_llm_cls):
        """Normal call should retrieve RAG context and call LLM."""
        # Mock vector store
        mock_vs = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Article 9: Risk management requirements."
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Article 14: Human oversight requirements."
        mock_vs.similarity_search.return_value = [mock_doc1, mock_doc2]
        mock_load_vs.return_value = mock_vs

        mock_llm_cls.return_value = self._mock_llm_client("Compliant narrative.")

        system = _make_fake_ai_system()
        result = generate_compliance_narrative(
            document_type=DocumentType.RISK_ASSESSMENT,
            ai_system=system,
            risk_assessment=None,
            company_name="Acme Corp",
            user_id=42,
        )

        assert result == "Compliant narrative."
        mock_load_vs.assert_called_once_with(user_id=42)
        mock_vs.similarity_search.assert_called_once()

    @patch("app.modules.llm.document_generator.LLMClient")
    @patch("app.modules.llm.document_generator.load_vector_store")
    def test_prompt_includes_system_name_and_type(self, mock_load_vs, mock_llm_cls):
        """The LLM should be called with a prompt containing system name and doc type."""
        captured_prompts = []

        def capture_call(prompt, system_prompt=None, max_tokens=2500):
            captured_prompts.append(prompt)
            return "Output"

        mock_llm_client = MagicMock()
        mock_llm_client.call.side_effect = capture_call
        mock_llm_cls.return_value = mock_llm_client

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_load_vs.return_value = mock_vs

        system = _make_fake_ai_system(
            name="Customer Churn Predictor",
            use_case="Predicting customer churn",
            sector="Banking",
        )
        result = generate_compliance_narrative(
            document_type=DocumentType.CONFORMITY_DECLARATION,
            ai_system=system,
            risk_assessment=None,
            company_name="Bank of Test",
            user_id=1,
        )

        assert len(captured_prompts) == 1
        assert "Customer Churn Predictor" in captured_prompts[0]
        assert "conformity_declaration" in captured_prompts[0]
        assert "Bank of Test" in captured_prompts[0]
        assert "Banking" in captured_prompts[0]

    @patch("app.modules.llm.document_generator.LLMClient")
    @patch("app.modules.llm.document_generator.load_vector_store")
    def test_risk_assessment_included_in_prompt(self, mock_load_vs, mock_llm_cls):
        """When risk_assessment is provided, its details should appear in the prompt."""
        captured_prompts = []

        def capture_call(prompt, system_prompt=None, max_tokens=2500):
            captured_prompts.append(prompt)
            return "Output"

        mock_llm_client = MagicMock()
        mock_llm_client.call.side_effect = capture_call
        mock_llm_cls.return_value = mock_llm_client

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_load_vs.return_value = mock_vs

        mock_risk_assessment = MagicMock()
        mock_risk_assessment.risk_level = RiskLevel.LIMITED
        mock_risk_assessment.findings = "No critical issues found."
        mock_risk_assessment.recommendations = "Continue monitoring."

        system = _make_fake_ai_system()
        generate_compliance_narrative(
            document_type=DocumentType.TECHNICAL_DOCUMENTATION,
            ai_system=system,
            risk_assessment=mock_risk_assessment,
            company_name="TestCo",
            user_id=None,
        )

        assert len(captured_prompts) == 1
        assert "No critical issues found" in captured_prompts[0]
        assert "Continue monitoring" in captured_prompts[0]

    @patch("app.modules.llm.document_generator.LLMClient")
    @patch("app.modules.llm.document_generator.load_vector_store")
    def test_none_company_name_uses_default(self, mock_load_vs, mock_llm_cls):
        """None company_name should render as 'Not Specified' in the prompt."""
        captured_prompts = []

        def capture_call(prompt, system_prompt=None, max_tokens=2500):
            captured_prompts.append(prompt)
            return "Output"

        mock_llm_client = MagicMock()
        mock_llm_client.call.side_effect = capture_call
        mock_llm_cls.return_value = mock_llm_client

        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_load_vs.return_value = mock_vs

        system = _make_fake_ai_system()
        generate_compliance_narrative(
            document_type=DocumentType.TECHNICAL_DOCUMENTATION,
            ai_system=system,
            risk_assessment=None,
            company_name=None,
            user_id=None,
        )

        assert len(captured_prompts) == 1
        assert "Not Specified" in captured_prompts[0]
