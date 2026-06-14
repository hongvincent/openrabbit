"""Shared pytest configuration for openrabbit.

Kept intentionally minimal. Module-specific fixtures live in each test
subtree. Unit tests MUST NOT make network calls or require live AWS/GitHub
credentials — use fakes/mocks/monkeypatch instead.
"""
