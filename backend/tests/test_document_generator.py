"""
Unit tests for backend/app/modules/llm/document_generator.py — generate_compliance_narrative.

Covers:
  - Normal path where load_vector_store returns docs and LLM returns content
  - FileNotFoundError fallback when vector store is uninitialized
  - Correct query construction
  - Risk assessment handling (None vs non-None)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models.ai_system import AISystem, RiskLevel
from app.models.document import DocumentType


class TestGenerateComplianceNarrative:
    """Tests for generate_compliance_narrative."""

    def _make_ai_system(
        self,
        name: str = "TestAI",
        use_case: str = "loan approval",
        sector: str = "finance",
        risk_level: RiskLevel = RiskLevel.HIGH,
    ) -> MagicMock:
        system = MagicMock(spec=AISystem)
        system.name = name
        system.use_case = use_case
        system.sector = sector
        system.risk_level = risk_level
        system.description = "A test AI system"
        system.version = "1.0"
        return system

    @patch("app.modules.llm.document_generator.load_vector_store")
    @patch("app.modules.llm.document_generator.LLMClient")
    def test_normal_path_returns_llm_content(
        self,
        mock_llm_client_cls: MagicMock,
        mock_load_vector_store: MagicMock,
    ) -> None:
        """When vector store and LLM succeed, the narrative is returned."""
        # Setup vector store mock
        mock_vs = MagicMock()
        doc = MagicMock()
        doc.page_content = "Article 9: Risk management system required."
        mock_vs.similarity_search.return_value = [doc]
        mock_load_vector_store.return_value = mock_vs

        # Setup LLM mock
        mock_client = MagicMock()
        mock_client.call.return_value = "Generated compliance document content."
        mock_llm_client_cls.return_value = mock_client

        from app.modules.llm.document_generator import generate_compliance_narrative

        system = self._make_ai_system()
        result = generate_compliance_narrative(
            document_type=DocumentType.TECHNICAL_REPORT,
            ai_system=system,
            risk_assessment=None,
            company_name="Test Corp",
            user_id=1,
        )

        assert result == "Generated compliance document content."
        mock_vs.similarity_search.assert_called_once()
        mock_client.call.assert_called_once()

    @patch("app.modules.llm.document_generator.load_vector_store")
    @patch("app.modules.llm.document_generator.LLMClient")
    def test_fallback_path_when_vector_store_not_initialized(
        self,
        mock_llm_client_cls: MagicMock,
        mock_load_vector_store: MagicMock,
    ) -> None:
        """When vector store raises FileNotFoundError, fallback rag_context is used."""
        mock_load_vector_store.side_effect = FileNotFoundError("No index found")

        mock_client = MagicMock()
        mock_client.call.return_value = "Fallback document."
        mock_llm_client_cls.return_value = mock_client

        from app.modules.llm.document_generator import generate_compliance_narrative

        system = self._make_ai_system()
        result = generate_compliance_narrative(
            document_type=DocumentType.TECHNICAL_REPORT,
            ai_system=system,
            risk_assessment=None,
            company_name="Test Corp",
            user_id=1,
        )

        assert result == "Fallback document."
        # Verify fallback context was in the prompt
        call_args = mock_client.call.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "No specific regulation context" in prompt

    @patch("app.modules.llm.document_generator.load_vector_store")
    @patch("app.modules.llm.document_generator.LLMClient")
    def test_query_includes_system_context(
        self,
        mock_llm_client_cls: MagicMock,
        mock_load_vector_store: MagicMock,
    ) -> None:
        """The vector store query should include system use_case and sector."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_load_vector_store.return_value = mock_vs

        mock_client = MagicMock()
        mock_client.call.return_value = "Doc"
        mock_llm_client_cls.return_value = mock_client

        from app.modules.llm.document_generator import generate_compliance_narrative

        system = self._make_ai_system(
            name="CreditScorer",
            use_case="credit scoring",
            sector="banking",
            risk_level=RiskLevel.HIGH,
        )
        generate_compliance_narrative(
            document_type=DocumentType.RISK_ASSESSMENT,
            ai_system=system,
            risk_assessment=None,
            company_name="BankCo",
            user_id=1,
        )

        # Verify the query includes use_case and sector
        call_args = mock_vs.similarity_search.call_args
        query = call_args[0][0] if call_args[0] else call_args.kwargs.get("query", "")
        assert "credit scoring" in query
        assert "banking" in query

    @patch("app.modules.llm.document_generator.load_vector_store")
    @patch("app.modules.llm.document_generator.LLMClient")
    def test_risk_assessment_included_in_prompt_when_present(
        self,
        mock_llm_client_cls: MagicMock,
        mock_load_vector_store: MagicMock,
    ) -> None:
        """When risk_assessment is provided, its details should be in the prompt."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_load_vector_store.return_value = mock_vs

        mock_client = MagicMock()
        mock_client.call.return_value = "Doc"
        mock_llm_client_cls.return_value = mock_client

        from app.modules.llm.document_generator import generate_compliance_narrative

        system = self._make_ai_system()

        # Create a mock risk assessment
        mock_assessment = MagicMock()
        mock_assessment.risk_level = RiskLevel.HIGH
        mock_assessment.findings = "Missing data governance procedures."
        mock_assessment.recommendations = "Implement data quality controls."

        generate_compliance_narrative(
            document_type=DocumentType.TECHNICAL_REPORT,
            ai_system=system,
            risk_assessment=mock_assessment,
            company_name="Test Corp",
            user_id=1,
        )

        call_args = mock_client.call.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "Missing data governance procedures" in prompt
        assert "Implement data quality controls" in prompt

    @patch("app.modules.llm.document_generator.load_vector_store")
    @patch("app.modules.llm.document_generator.LLMClient")
    def test_company_name_defaults_when_none(
        self,
        mock_llm_client_cls: MagicMock,
        mock_load_vector_store: MagicMock,
    ) -> None:
        """When company_name is None, 'Not Specified' is used in the prompt."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_load_vector_store.return_value = mock_vs

        mock_client = MagicMock()
        mock_client.call.return_value = "Doc"
        mock_llm_client_cls.return_value = mock_client

        from app.modules.llm.document_generator import generate_compliance_narrative

        system = self._make_ai_system()
        generate_compliance_narrative(
            document_type=DocumentType.TECHNICAL_REPORT,
            ai_system=system,
            risk_assessment=None,
            company_name=None,
            user_id=1,
        )

        call_args = mock_client.call.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "Not Specified" in prompt
