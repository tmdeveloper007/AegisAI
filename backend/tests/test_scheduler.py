"""
Unit tests for backend/app/tasks/scheduler.py —
snapshot_compliance_scores and send_reassessment_reminders.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app.tasks.scheduler import snapshot_compliance_scores, send_reassessment_reminders


class TestSnapshotComplianceScores:
    """Tests for snapshot_compliance_scores()."""

    @patch("app.tasks.scheduler.SessionLocal")
    @patch("app.tasks.scheduler.ComplianceSnapshot")
    def test_creates_snapshot_for_system_with_score(self, mock_snapshot_cls, mock_session_local_cls):
        """When a system has a compliance_score, a ComplianceSnapshot row is created."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db

        mock_system = MagicMock()
        mock_system.compliance_score = 85.5
        mock_system.compliance_status.value = "compliant"
        mock_system.risk_level.value = "high"
        mock_system.id = 1

        mock_db.query.return_value.all.return_value = [mock_system]

        snapshot_compliance_scores()

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch("app.tasks.scheduler.SessionLocal")
    @patch("app.tasks.scheduler.ComplianceSnapshot")
    def test_skips_system_with_none_score(self, mock_snapshot_cls, mock_session_local_cls):
        """Systems with None compliance_score are skipped and no snapshot is created."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db

        mock_system = MagicMock()
        mock_system.compliance_score = None
        mock_system.id = 2

        mock_db.query.return_value.all.return_value = [mock_system]

        snapshot_compliance_scores()

        # No add calls when score is None
        mock_db.add.assert_not_called()
        mock_db.commit.assert_called_once()

    @patch("app.tasks.scheduler.SessionLocal")
    @patch("app.tasks.scheduler.ComplianceSnapshot")
    def test_handles_empty_database(self, mock_snapshot_cls, mock_session_local_cls):
        """When no systems exist, the function runs without error."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db
        mock_db.query.return_value.all.return_value = []

        # Should not raise
        snapshot_compliance_scores()

        mock_db.commit.assert_called_once()

    @patch("app.tasks.scheduler.SessionLocal")
    def test_rollback_on_exception(self, mock_session_local_cls):
        """If an exception occurs, the transaction is rolled back and the error is logged."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db
        mock_db.query.return_value.all.side_effect = RuntimeError("db error")

        # Function catches exceptions internally; no exception propagates
        snapshot_compliance_scores()

        mock_db.rollback.assert_called_once()
        mock_db.close.assert_called_once()


class TestSendReassessmentReminders:
    """Tests for send_reassessment_reminders()."""

    @patch("app.tasks.scheduler.SessionLocal")
    def test_sends_reminder_for_expiring_assessment(self, mock_session_local_cls):
        """Assessment expiring within 30 days triggers a notification."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db

        # Assessment expiring in 15 days
        expiring_date = datetime.utcnow() + timedelta(days=15)
        mock_assessment = MagicMock()
        mock_assessment.valid_until = expiring_date
        mock_assessment.id = 10
        mock_assessment.ai_system_id = 5

        mock_system = MagicMock()
        mock_system.name = "Test AI System"
        mock_system.owner_id = 1

        mock_db.query.return_value.filter.return_value.all.return_value = [mock_assessment]
        mock_db.query.return_value.filter.return_value.first.return_value = mock_system

        send_reassessment_reminders()

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch("app.tasks.scheduler.SessionLocal")
    def test_does_not_send_for_already_expired(self, mock_session_local_cls):
        """Assessments that have already expired do not trigger reminders."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db

        # The query returns empty because expired assessments fail the valid_until > now filter
        mock_db.query.return_value.filter.return_value.all.return_value = []

        send_reassessment_reminders()

        # No notification added when no expiring assessments are returned
        mock_db.add.assert_not_called()

    @patch("app.tasks.scheduler.SessionLocal")
    def test_skips_missing_system(self, mock_session_local_cls):
        """If the associated AI system is deleted, no notification is sent."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db

        expiring_date = datetime.utcnow() + timedelta(days=20)
        mock_assessment = MagicMock()
        mock_assessment.valid_until = expiring_date
        mock_assessment.ai_system_id = 99

        mock_db.query.return_value.filter.return_value.all.return_value = [mock_assessment]
        mock_db.query.return_value.filter.return_value.first.return_value = None

        send_reassessment_reminders()

        mock_db.add.assert_not_called()
        mock_db.commit.assert_called_once()

    @patch("app.tasks.scheduler.SessionLocal")
    def test_handles_empty_results(self, mock_session_local_cls):
        """When no assessments are expiring, the function runs without error."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = []

        send_reassessment_reminders()

        mock_db.commit.assert_called_once()

    @patch("app.tasks.scheduler.SessionLocal")
    def test_rollback_on_exception(self, mock_session_local_cls):
        """If an exception occurs, the transaction is rolled back and the error is logged."""
        mock_db = MagicMock()
        mock_session_local_cls.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.side_effect = RuntimeError("db error")

        # Function catches exceptions internally; no exception propagates
        send_reassessment_reminders()

        mock_db.rollback.assert_called_once()
        mock_db.close.assert_called_once()
