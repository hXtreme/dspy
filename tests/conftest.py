"""Shared pytest fixtures for the dspy.graph test suite.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

import pytest

from dspy.graph.storage import (
    AdjecencyMatrixGraphStorage,
    LinkedListGraphStorage,
)
from dspy.graph.types import GraphStorageProvider

_STORAGE_PROVIDERS = [
    pytest.param(AdjecencyMatrixGraphStorage, id="adjacency-matrix"),
    pytest.param(LinkedListGraphStorage, id="linked-list"),
]


@pytest.fixture(params=_STORAGE_PROVIDERS)
def storage(request: pytest.FixtureRequest) -> GraphStorageProvider:
    """Parametrize over every storage backend."""
    return request.param
