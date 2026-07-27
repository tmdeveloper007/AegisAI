"""
Unit tests for validate_password_strength in security.py.

Covers the bcrypt-72-byte limit, weak-password rejection cases, and the
valid-password happy path.
"""

import pytest
from app.core.security import validate_password_strength


class TestValidatePasswordStrength:
    def test_valid_password_returns_password(self):
        """A password meeting all requirements is returned unchanged."""
        result = validate_password_strength("ValidPass123!")
        assert result == "ValidPass123!"

    def test_valid_password_exactly_8_chars(self):
        """A password at exactly 8 characters with all requirements passes."""
        result = validate_password_strength("Ab1!xxxx")
        assert result == "Ab1!xxxx"

    def test_password_exceeding_72_bytes_raises(self):
        """Passwords that encode to more than 72 UTF-8 bytes are rejected."""
        # The character 'e' with acute accent encodes to 2 bytes in UTF-8.
        # 36 accented characters + "A1!" = 36*2 + 3 = 75 bytes, exceeds 72.
        long_password = ("\u00e9" * 36) + "A1!"
        with pytest.raises(ValueError, match="72 bytes"):
            validate_password_strength(long_password)

    def test_password_shorter_than_8_chars_raises(self):
        """Passwords under 8 characters are rejected with a descriptive message."""
        with pytest.raises(ValueError, match="at least 8 characters"):
            validate_password_strength("Ab1!xyz")

    def test_password_missing_uppercase_raises(self):
        """Passwords without an uppercase letter are rejected."""
        with pytest.raises(ValueError, match="at least one uppercase letter"):
            validate_password_strength("lowercase123!")

    def test_password_missing_digit_raises(self):
        """Passwords without a digit are rejected."""
        with pytest.raises(ValueError, match="at least one digit"):
            validate_password_strength("AllUpperCase!")

    def test_password_missing_special_char_raises(self):
        """Passwords without a special character are rejected."""
        with pytest.raises(ValueError, match="at least one special character"):
            validate_password_strength("ValidPass1234")

    def test_multiple_weaknesses_raises_with_combined_message(self):
        """When multiple requirements are missing, the ValueError lists them all."""
        with pytest.raises(ValueError) as exc_info:
            validate_password_strength("weak")
        msg = str(exc_info.value)
        assert "at least 8 characters" in msg
        assert "at least one uppercase letter" in msg
        assert "at least one digit" in msg
        assert "at least one special character" in msg
