"""Shared pytest fixtures for the dspy.graph test suite.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

import pytest

from dspy.graph.storage import (
    AdjacencyMatrixGraphStorage,
    LinkedListGraphStorage,
)
from dspy.graph.types import GraphStorage

_STORAGE_PROVIDERS = [
    pytest.param(AdjacencyMatrixGraphStorage, id="adjacency-matrix"),
    pytest.param(LinkedListGraphStorage, id="linked-list"),
]


@pytest.fixture(params=_STORAGE_PROVIDERS)
def storage(request: pytest.FixtureRequest) -> type[GraphStorage]:
    """Parametrize over every storage backend."""
    return request.param
