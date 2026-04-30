#!/usr/bin/env python3
"""
Test suite for ast-grep YAML rule validator.

Usage:
    uv run python test_validate_astgrep_rules.py
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest

# Import the validator
from validate_astgrep_rules import (
    RuleValidator,
    ValidationResult,
    VALID_LANGUAGES,
)


# Test fixtures
@pytest.fixture
def validator():
    """Create a validator instance without sg checking."""
    return RuleValidator(auto_fix=False, generate_tests=False, check_sg_available=False)


@pytest.fixture
def auto_fix_validator():
    """Create a validator with auto-fix enabled."""
    return RuleValidator(auto_fix=True, generate_tests=False, check_sg_available=False)


@pytest.fixture
def temp_rule_file(tmp_path):
    """Create a temporary rule file."""
    def _create_rule(content: str) -> Path:
        rule_file = tmp_path / "test-rule.yml"
        rule_file.write_text(content, encoding='utf-8')
        return rule_file
    return _create_rule


# Valid rule examples
VALID_RULE_BASIC = """id: test-rule-basic
language: JavaScript
rule:
  pattern: console.log($A)
"""

VALID_RULE_COMPLETE = """id: test-rule-complete
language: TypeScript
severity: warning
message: Avoid console.log in production
rule:
  pattern: console.log($A)
fix: logger.info($A)
"""

VALID_RULE_COMPLEX = """id: test-rule-complex
language: Python
severity: error
message: Use list comprehension instead of map/filter
rule:
  any:
    - pattern: map($FUNC, $LIST)
    - pattern: filter($FUNC, $LIST)
"""

# Invalid rule examples
INVALID_RULE_MISSING_ID = """language: JavaScript
rule:
  pattern: console.log($A)
"""

INVALID_RULE_MISSING_LANGUAGE = """id: test-missing-lang
rule:
  pattern: console.log($A)
"""

INVALID_RULE_MISSING_RULE = """id: test-missing-rule
language: JavaScript
"""

INVALID_RULE_WRONG_LANGUAGE = """id: test-wrong-lang
language: javascript
rule:
  pattern: console.log($A)
"""

INVALID_RULE_EMPTY_RULE = """id: test-empty-rule
language: JavaScript
rule: {}
"""

INVALID_YAML_SYNTAX = """id: test-bad-yaml
language: JavaScript
rule:
  pattern: console.log($A
  - this is broken
"""


class TestRuleValidator:
    """Test RuleValidator class."""

    def test_init(self):
        """Test validator initialization."""
        validator = RuleValidator()
        assert validator.auto_fix is False
        assert validator.generate_tests is False

        validator = RuleValidator(auto_fix=True, generate_tests=True)
        assert validator.auto_fix is True
        assert validator.generate_tests is True

    def test_validate_yaml_syntax_valid(self, validator, temp_rule_file):
        """Test YAML syntax validation with valid YAML."""
        rule_file = temp_rule_file(VALID_RULE_BASIC)
        valid, data, error = validator.validate_yaml_syntax(rule_file)

        assert valid is True
        assert data is not None
        assert error is None
        assert 'id' in data
        assert data['id'] == 'test-rule-basic'

    def test_validate_yaml_syntax_invalid(self, validator, temp_rule_file):
        """Test YAML syntax validation with invalid YAML."""
        rule_file = temp_rule_file(INVALID_YAML_SYNTAX)
        valid, data, error = validator.validate_yaml_syntax(rule_file)

        assert valid is False
        assert data is None
        assert error is not None
        assert 'YAML syntax error' in error

    def test_validate_yaml_syntax_empty(self, validator, temp_rule_file):
        """Test YAML syntax validation with empty file."""
        rule_file = temp_rule_file("")
        valid, data, error = validator.validate_yaml_syntax(rule_file)

        assert valid is False
        assert error is not None
        assert 'empty' in error.lower()

    def test_validate_required_fields_valid(self, validator):
        """Test required fields validation with valid data."""
        data = {
            'id': 'test',
            'language': 'JavaScript',
            'rule': {'pattern': 'test'}
        }
        errors = validator.validate_required_fields(data)
        assert len(errors) == 0

    def test_validate_required_fields_missing_id(self, validator):
        """Test required fields validation with missing id."""
        data = {
            'language': 'JavaScript',
            'rule': {'pattern': 'test'}
        }
        errors = validator.validate_required_fields(data)
        assert len(errors) == 1
        assert 'id' in errors[0].lower()

    def test_validate_required_fields_missing_multiple(self, validator):
        """Test required fields validation with multiple missing fields."""
        data = {'id': 'test'}
        errors = validator.validate_required_fields(data)
        assert len(errors) == 1
        assert 'language' in errors[0]
        assert 'rule' in errors[0]

    def test_validate_required_fields_wrong_type(self, validator):
        """Test required fields validation with wrong types."""
        data = {
            'id': 123,  # Should be string
            'language': 'JavaScript',
            'rule': 'not a dict'  # Should be dict
        }
        errors = validator.validate_required_fields(data)
        assert len(errors) >= 2
        assert any('id' in e for e in errors)
        assert any('rule' in e for e in errors)

    def test_validate_language_valid(self, validator):
        """Test language validation with valid languages."""
        for lang in VALID_LANGUAGES:
            valid, error, correction = validator.validate_language(lang)
            assert valid is True, f"Language {lang} should be valid"
            assert error is None
            assert correction is None

    def test_validate_language_invalid_with_alias(self, validator):
        """Test language validation with known aliases."""
        test_cases = [
            ('javascript', 'JavaScript'),
            ('typescript', 'TypeScript'),
            ('python', 'Python'),
            ('rust', 'Rust'),
        ]

        for alias, expected in test_cases:
            valid, error, correction = validator.validate_language(alias)
            assert valid is False
            assert error is not None
            assert correction == expected

    def test_validate_language_invalid_with_suggestion(self, validator):
        """Test language validation with fuzzy matching."""
        valid, error, correction = validator.validate_language('Javascrpt')
        assert valid is False
        assert error is not None
        assert 'JavaScript' in error

    def test_validate_language_invalid_no_match(self, validator):
        """Test language validation with no close match."""
        valid, error, correction = validator.validate_language('Cobol')
        assert valid is False
        assert error is not None
        assert 'Valid options' in error

    def test_normalize_yaml_key_order(self, validator):
        """Test YAML normalization maintains key order."""
        data = {
            'fix': 'some fix',
            'message': 'some message',
            'rule': {'pattern': 'test'},
            'language': 'JavaScript',
            'id': 'test-id',
            'extra_field': 'value',
        }

        normalized = validator.normalize_yaml(data)
        keys = list(normalized.keys())

        # Check that standard keys come first in order
        assert keys[0] == 'id'
        assert keys[1] == 'language'
        # message, rule, fix should come before extra_field
        assert keys.index('extra_field') > keys.index('rule')

    def test_validate_file_valid_basic(self, validator, temp_rule_file):
        """Test file validation with valid basic rule."""
        rule_file = temp_rule_file(VALID_RULE_BASIC)
        result = validator.validate_file(rule_file)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.rule_id == 'test-rule-basic'

    def test_validate_file_valid_complete(self, validator, temp_rule_file):
        """Test file validation with complete valid rule."""
        rule_file = temp_rule_file(VALID_RULE_COMPLETE)
        result = validator.validate_file(rule_file)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.rule_id == 'test-rule-complete'

    def test_validate_file_missing_id(self, validator, temp_rule_file):
        """Test file validation with missing id."""
        rule_file = temp_rule_file(INVALID_RULE_MISSING_ID)
        result = validator.validate_file(rule_file)

        assert result.is_valid is False
        assert any('id' in e.lower() for e in result.errors)

    def test_validate_file_missing_language(self, validator, temp_rule_file):
        """Test file validation with missing language."""
        rule_file = temp_rule_file(INVALID_RULE_MISSING_LANGUAGE)
        result = validator.validate_file(rule_file)

        assert result.is_valid is False
        assert any('language' in e.lower() for e in result.errors)

    def test_validate_file_missing_rule(self, validator, temp_rule_file):
        """Test file validation with missing rule."""
        rule_file = temp_rule_file(INVALID_RULE_MISSING_RULE)
        result = validator.validate_file(rule_file)

        assert result.is_valid is False
        assert any('rule' in e.lower() for e in result.errors)

    def test_validate_file_wrong_language(self, validator, temp_rule_file):
        """Test file validation with incorrect language case."""
        rule_file = temp_rule_file(INVALID_RULE_WRONG_LANGUAGE)
        result = validator.validate_file(rule_file)

        assert result.is_valid is False
        assert any('language' in e.lower() for e in result.errors)
        assert any('JavaScript' in s for s in result.suggestions)

    def test_validate_file_empty_rule(self, validator, temp_rule_file):
        """Test file validation with empty rule object."""
        rule_file = temp_rule_file(INVALID_RULE_EMPTY_RULE)
        result = validator.validate_file(rule_file)

        assert result.is_valid is False
        assert any('empty' in e.lower() for e in result.errors)

    def test_auto_fix_basic(self, auto_fix_validator, temp_rule_file):
        """Test auto-fix functionality."""
        # Create rule with wrong language case
        rule_content = """id: test-auto-fix
language: javascript
rule:
  pattern: console.log($A)
message: Test message
fix: logger.log($A)
"""
        rule_file = temp_rule_file(rule_content)
        result = auto_fix_validator.validate_file(rule_file)

        assert result.auto_fixed is True
        assert len(result.suggestions) > 0

        # Read back and verify language was corrected
        content = rule_file.read_text(encoding='utf-8')
        assert 'language: JavaScript' in content
        assert 'language: javascript' not in content

    def test_auto_fix_key_order(self, auto_fix_validator, temp_rule_file):
        """Test auto-fix normalizes key order."""
        rule_content = """fix: logger.log($A)
message: Test message
rule:
  pattern: console.log($A)
language: JavaScript
id: test-key-order
"""
        rule_file = temp_rule_file(rule_content)
        result = auto_fix_validator.validate_file(rule_file)

        assert result.auto_fixed is True

        # Read back and verify key order
        content = rule_file.read_text(encoding='utf-8')
        lines = [line for line in content.split('\n') if line.strip()]

        # id should come first
        assert lines[0].startswith('id:')
        # language should come second
        assert lines[1].startswith('language:')

    def test_auto_fix_trailing_newline(self, auto_fix_validator, temp_rule_file):
        """Test auto-fix adds trailing newline."""
        rule_content = VALID_RULE_BASIC.rstrip('\n')
        rule_file = temp_rule_file(rule_content)

        # Verify no trailing newline initially
        assert not rule_file.read_text(encoding='utf-8').endswith('\n\n')

        result = auto_fix_validator.validate_file(rule_file)

        assert result.auto_fixed is True

        # Verify trailing newline was added
        content = rule_file.read_text(encoding='utf-8')
        assert content.endswith('\n')

    def test_generate_test_file(self, validator, temp_rule_file):
        """Test test file generation."""
        rule_file = temp_rule_file(VALID_RULE_BASIC)

        from ruamel.yaml import YAML
        yaml = YAML()
        with open(rule_file, 'r', encoding='utf-8') as f:
            rule_data = yaml.load(f)

        test_file = validator.generate_test_file(rule_file, rule_data)

        assert test_file is not None
        assert test_file.exists()
        assert test_file.name == 'test-rule-basic-test.yml'

        # Verify test file structure
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = yaml.load(f)

        assert 'id' in test_data
        assert 'language' in test_data
        assert 'valid' in test_data
        assert 'invalid' in test_data
        assert len(test_data['valid']) > 0
        assert len(test_data['invalid']) > 0

    def test_validate_directory(self, validator, tmp_path):
        """Test directory validation."""
        # Create multiple rule files
        rules_dir = tmp_path / 'rules'
        rules_dir.mkdir()

        (rules_dir / 'rule1.yml').write_text(VALID_RULE_BASIC, encoding='utf-8')
        (rules_dir / 'rule2.yml').write_text(VALID_RULE_COMPLETE, encoding='utf-8')
        (rules_dir / 'rule3.yml').write_text(INVALID_RULE_MISSING_ID, encoding='utf-8')

        report = validator.validate_directory(rules_dir)

        assert report.total_files == 3
        assert report.valid_files == 2
        assert report.invalid_files == 1

    def test_validate_directory_duplicate_ids(self, validator, tmp_path):
        """Test duplicate ID detection."""
        rules_dir = tmp_path / 'rules'
        rules_dir.mkdir()

        # Create two rules with same ID
        rule_dup1 = """id: duplicate-id
language: JavaScript
rule:
  pattern: test1
"""
        rule_dup2 = """id: duplicate-id
language: TypeScript
rule:
  pattern: test2
"""

        (rules_dir / 'dup1.yml').write_text(rule_dup1, encoding='utf-8')
        (rules_dir / 'dup2.yml').write_text(rule_dup2, encoding='utf-8')

        report = validator.validate_directory(rules_dir)

        assert len(report.duplicate_ids) == 1
        assert report.duplicate_ids[0][0] == 'duplicate-id'
        assert len(report.duplicate_ids[0][1]) == 2

    def test_validation_report_to_dict(self, validator, temp_rule_file):
        """Test validation report JSON serialization."""
        rule_file = temp_rule_file(VALID_RULE_BASIC)
        result = validator.validate_file(rule_file)

        from validate_astgrep_rules import ValidationReport
        report = ValidationReport(
            total_files=1,
            valid_files=1,
            invalid_files=0,
            results=[result]
        )

        report_dict = report.to_dict()

        assert 'summary' in report_dict
        assert 'results' in report_dict
        assert report_dict['summary']['total_files'] == 1
        assert report_dict['summary']['valid_files'] == 1

        # Verify JSON serializable
        json_str = json.dumps(report_dict)
        assert len(json_str) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file(self, validator, tmp_path):
        """Test validation of non-existent file."""
        nonexistent = tmp_path / 'does-not-exist.yml'
        result = validator.validate_file(nonexistent)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_binary_file(self, validator, tmp_path):
        """Test validation of binary file."""
        binary_file = tmp_path / 'binary.yml'
        binary_file.write_bytes(b'\x00\x01\x02\x03')

        result = validator.validate_file(binary_file)

        assert result.is_valid is False

    def test_unicode_content(self, validator, tmp_path):
        """Test validation with Unicode content."""
        rule_content = """id: unicode-test
language: JavaScript
message: 你好世界 🌍
rule:
  pattern: console.log($A)
"""
        rule_file = tmp_path / 'unicode.yml'
        rule_file.write_text(rule_content, encoding='utf-8')

        result = validator.validate_file(rule_file)

        assert result.is_valid is True
        assert result.rule_id == 'unicode-test'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
