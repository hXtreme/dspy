"""Graph data structures with pluggable storage backends.

This package provides abstract graph protocols and concrete implementations
such as :class:`AsymmetricMultiDiGraph`, along with multiple storage backends
(adjacency matrix and linked list).

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

from dspy.graph.impl import AsymmetricMultiDiGraph

__all__ = [
    "AsymmetricMultiDiGraph",
]
