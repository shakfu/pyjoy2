"""
Pytest configuration and shared fixtures.
"""

import pytest
from pyjoy2.core import Stack, WORDS


@pytest.fixture
def stack():
    """Fresh stack for each test."""
    return Stack()


@pytest.fixture
def populated_stack():
    """Stack with some initial values."""
    s = Stack()
    s.push(1, 2, 3)
    return s


@pytest.fixture(autouse=True)
def reset_words():
    """
    Save and restore WORDS state around each test.
    This prevents test-defined words from leaking between tests.
    """
    original_words = dict(WORDS)
    yield
    WORDS.clear()
    WORDS.update(original_words)
