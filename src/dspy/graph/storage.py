"""Graph storage backend implementations.

Provides concrete :class:`GraphStorage` implementations using different
underlying data structures:

* :data:`AdjecencyMatrixGraphStorage` -- backed by a 2-D matrix (uses NumPy
  when available, falls back to nested Python lists).
* :data:`LinkedListGraphStorage` -- backed by per-vertex adjacency lists.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

import itertools
from collections.abc import Iterable
from typing import Self, cast

from dspy.graph.types import EdgeID, EdgeMonad, GraphStorage, VertexID


class IdentityEdgeMonad(EdgeMonad):
    """Edge monad that retains only the most recent edge identifier.

    Suitable for simple graphs where at most one edge exists between any
    ordered vertex pair.  Each :meth:`chain` call replaces the previously
    stored :class:`EdgeID`.
    """

    _e: EdgeID

    def __init__(self, e: EdgeID) -> None:
        """Initialise the monad with a single edge identifier.

        :param e: The edge identifier to store.
        """
        self._e = e

    @classmethod
    def bind(cls, e: EdgeID) -> Self:
        """Create a new monad containing a single edge identifier.

        :param e: The initial edge identifier.
        :return: A fresh :class:`IdentityEdgeMonad` wrapping *e*.
        """
        return cls(e)

    def chain(self, e: EdgeID) -> Self:
        """Replace the stored edge identifier with *e*.

        :param e: The new edge identifier that supersedes the previous one.
        :return: ``self``, updated in place.
        """
        self._e = e
        return self

    def unwrap(self) -> Iterable[EdgeID]:
        """Return the single stored edge identifier as a one-element tuple.

        :return: A tuple containing the current :class:`EdgeID`.
        """
        return (self._e,)


try:
    import numpy as np

    class _AdjecencyMatrixGraphStorage[EM: EdgeMonad](GraphStorage[EM]):
        _n: int
        _edge_monad: type[EM]
        _vertices: np.ndarray[tuple[int], np.dtype[np.bool]]
        _matrix: np.ndarray[tuple[int, int], np.dtype[np.object_]]

        def __init__(self, n: int, edge_monad: type[EM]) -> None:
            self._edge_monad = edge_monad
            self._n = n
            self._vertices = np.ones((n,)).astype(bool)
            self._matrix = np.full((n, n), None, dtype="object")

        def update_edge(self, u: VertexID, v: VertexID, value: EdgeID) -> EM:
            new_edge = self._update_edge(value, self._matrix[u][v])
            self._matrix[u, v] = new_edge
            return new_edge

        def vertices(self) -> Iterable[VertexID]:
            return np.arange(self._n)[self._vertices]

        def _edges(self) -> Iterable[tuple[VertexID, VertexID, EM]]:
            for u in self.vertices():
                yield from self._outgoing_edges(u)

        def _incoming_edges(
            self,
            v: VertexID,
        ) -> Iterable[tuple[VertexID, VertexID, EM]]:
            column: np.ndarray[tuple[int], np.dtype[np.object_]] = self._matrix[:, v]
            mask = (
                cast(
                    "np.ndarray[tuple[int], np.dtype[np.bool_]]",
                    column.astype(bool),
                )
                & self._vertices
            )
            destination: Iterable[VertexID] = cast(
                "np.ndarray[tuple[int], np.dtype[np.int64]]",
                np.full((self._n,), v, dtype=int)[mask],
            )
            source: Iterable[VertexID] = cast(
                "np.ndarray[tuple[int], np.dtype[np.int64]]",
                np.arange(self._n)[mask],
            )
            values: Iterable[EM] = column[mask]
            return zip(source, destination, values, strict=True)

        def _outgoing_edges(
            self,
            u: VertexID,
        ) -> Iterable[tuple[VertexID, VertexID, EM]]:
            row: np.ndarray[tuple[int], np.dtype[np.object_]] = self._matrix[u]
            mask = (
                cast(
                    "np.ndarray[tuple[int], np.dtype[np.bool_]]",
                    row.astype(bool),
                )
                & self._vertices
            )
            source: Iterable[VertexID] = cast(
                "np.ndarray[tuple[int], np.dtype[np.int64]]",
                np.full((self._n,), u, dtype=int)[mask],
            )
            destination: Iterable[VertexID] = cast(
                "np.ndarray[tuple[int], np.dtype[np.int64]]",
                np.arange(self._n)[mask],
            )
            values: Iterable[EM] = row[mask]
            return zip(source, destination, values, strict=True)
except ImportError:

    class _AdjecencyMatrixGraphStorage[EM: EdgeMonad](GraphStorage[EM]):
        _n: int
        _edge_monad: type[EM]
        _vertices: set[VertexID]
        _matrix: list[list[EM | None]]

        def __init__(self, n: int, edge_monad: type[EM]) -> None:
            self._edge_monad = edge_monad
            self._n = n
            self._vertices = set(map(VertexID, range(n)))
            self._matrix = [[None] * n for _ in range(n)]

        def update_edge(self, u: VertexID, v: VertexID, value: EdgeID) -> EM:
            new_edge = self._update_edge(value, self._matrix[u][v])
            self._matrix[u][v] = new_edge
            return new_edge

        def vertices(self) -> Iterable[VertexID]:
            return frozenset(self._vertices)

        def _edges(self) -> Iterable[tuple[VertexID, VertexID, EM]]:
            for u in self.vertices():
                yield from self._outgoing_edges(u)

        def _incoming_edges(
            self,
            v: VertexID,
        ) -> Iterable[tuple[VertexID, VertexID, EM]]:
            return [
                (u, v, val)
                for u in self._vertices
                if (val := self._matrix[u][v]) is not None
            ]

        def _outgoing_edges(
            self,
            u: VertexID,
        ) -> Iterable[tuple[VertexID, VertexID, EM]]:
            return [
                (u, v, val)
                for v in self._vertices
                if (val := self._matrix[u][v]) is not None
            ]


class _LinkedListGraphStorage[EM: EdgeMonad](GraphStorage[EM]):
    _n: int
    _edge_monad: type[EM]
    _vertices: set[VertexID]
    _matrix: list[list[tuple[VertexID, EM]]]

    def __init__(self, n: int, edge_monad: type[EM]) -> None:
        self._edge_monad = edge_monad
        self._n = n
        self._vertices = set(map(VertexID, range(n)))
        self._matrix = [[] for _ in range(n)]

    def update_edge(self, u: VertexID, v: VertexID, value: EdgeID) -> EM:
        matching_edge = (
            (idx, edge) for idx, edge in enumerate(self._matrix[u]) if edge[0] == v
        )
        idx, (_, edge_monad) = next(matching_edge, (-1, (v, None)))
        new_edge = (v, self._update_edge(value, edge_monad))

        if idx < 0:
            self._matrix[u].append(new_edge)
            return new_edge[1]

        self._matrix[u][idx] = new_edge
        return new_edge[1]

    def vertices(self) -> Iterable[VertexID]:
        return frozenset(self._vertices)

    def _edges(self) -> Iterable[tuple[VertexID, VertexID, EM]]:
        for u, u_v_edges in enumerate(self._matrix):
            if u not in self._vertices:
                continue
            yield from ((u, v, value) for v, value in u_v_edges)

    def _incoming_edges(self, v: VertexID) -> Iterable[tuple[VertexID, VertexID, EM]]:
        matching_edges = itertools.chain(
            *(
                ((VertexID(u), v, value) for (w, value) in edges if v == w)
                for u, edges in enumerate(self._matrix)
            ),
        )
        return list(matching_edges)

    def _outgoing_edges(self, u: VertexID) -> Iterable[tuple[VertexID, VertexID, EM]]:
        return [(u, v, val) for v, val in self._matrix[u]]


AdjecencyMatrixGraphStorage: type[GraphStorage] = _AdjecencyMatrixGraphStorage
LinkedListGraphStorage: type[GraphStorage] = _LinkedListGraphStorage
