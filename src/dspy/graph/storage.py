"""Graph storage backend implementations.

Provides concrete :class:`GraphStorage` implementations using different
underlying data structures:

* :data:`AdjecencyMatrixGraphStorage` -- backed by a 2-D matrix (uses NumPy
  when available, falls back to nested Python lists).
* :data:`LinkedListGraphStorage` -- backed by per-vertex adjacency lists.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

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

    def __str__(self) -> str:
        """Return the string form of the single stored edge identifier.

        :return: ``str(self._e)``.
        """
        return str(self._e)


try:
    import numpy as np

    class _AdjecencyMatrixGraphStorage[EM: EdgeMonad](GraphStorage[EM]):
        """NumPy-backed adjacency-matrix storage for an ``n``-vertex graph.

        Holds an ``n x n`` object matrix whose ``[u, v]`` cell is either
        ``None`` (no edge) or an :class:`EdgeMonad` accumulating every
        :class:`EdgeID` between ``u`` and ``v``.  A separate boolean mask
        tracks which vertex slots are active.

        Selected when NumPy is importable; the pure-Python class with the
        same name below is used otherwise.
        """

        _n: int
        _edge_monad: type[EM]
        _vertices: np.ndarray[tuple[int], np.dtype[np.bool]]
        _matrix: np.ndarray[tuple[int, int], np.dtype[np.object_]]

        def __init__(self, n: int, edge_monad: type[EM]) -> None:
            """Allocate the vertex mask and an empty ``n x n`` matrix.

            :param n: Number of vertex slots to reserve.  All slots start
                active.
            :param edge_monad: The :class:`EdgeMonad` class used to compose
                edge identifiers between every vertex pair.
            """
            self._edge_monad = edge_monad
            self._n = n
            self._vertices = np.ones((n,)).astype(bool)
            self._matrix = np.full((n, n), None, dtype="object")

        def update_edge(self, u: VertexID, v: VertexID, value: EdgeID) -> EM:
            """Insert or update the edge monad at cell ``(u, v)``.

            If the cell is empty a fresh monad is bound from *value*;
            otherwise *value* is chained onto the existing monad.

            :param u: Source vertex identifier.
            :param v: Destination vertex identifier.
            :param value: The edge identifier to store at ``(u, v)``.
            :return: The edge monad now stored at ``(u, v)``.
            """
            new_edge = self._update_edge(value, self._matrix[u][v])
            self._matrix[u, v] = new_edge
            return new_edge

        def vertices(self) -> Iterable[VertexID]:
            """Return the identifiers of all currently active vertex slots.

            :return: An iterable of :class:`VertexID` values, one per
                slot whose entry in the vertex mask is ``True``.
            """
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
        """Pure-Python adjacency-matrix storage used when NumPy is absent.

        Mirrors the NumPy variant's contract: an ``n x n`` matrix whose
        cells are either ``None`` or an :class:`EdgeMonad`, plus a set of
        active vertex identifiers.  The matrix is a nested ``list``;
        ``vertices()`` returns the active set as a ``frozenset``.
        """

        _n: int
        _edge_monad: type[EM]
        _vertices: set[VertexID]
        _matrix: list[list[EM | None]]

        def __init__(self, n: int, edge_monad: type[EM]) -> None:
            """Allocate the active-vertex set and an empty matrix.

            :param n: Number of vertex slots to reserve.  Identifiers
                ``0`` through ``n - 1`` are all marked active.
            :param edge_monad: The :class:`EdgeMonad` class used to compose
                edge identifiers between every vertex pair.
            """
            self._edge_monad = edge_monad
            self._n = n
            self._vertices = set(map(VertexID, range(n)))
            self._matrix = [[None] * n for _ in range(n)]

        def update_edge(self, u: VertexID, v: VertexID, value: EdgeID) -> EM:
            """Insert or update the edge monad at cell ``(u, v)``.

            If the cell is empty a fresh monad is bound from *value*;
            otherwise *value* is chained onto the existing monad.

            :param u: Source vertex identifier.
            :param v: Destination vertex identifier.
            :param value: The edge identifier to store at ``(u, v)``.
            :return: The edge monad now stored at ``(u, v)``.
            """
            new_edge = self._update_edge(value, self._matrix[u][v])
            self._matrix[u][v] = new_edge
            return new_edge

        def vertices(self) -> Iterable[VertexID]:
            """Return a snapshot of the currently active vertex set.

            :return: A :class:`frozenset` of active :class:`VertexID`
                values.
            """
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
    """Adjacency-list storage for an ``n``-vertex graph.

    Each vertex ``u`` owns a Python list of ``(destination, EdgeMonad)``
    pairs.  Inserting an edge updates the existing pair for the same
    destination when present, otherwise appends a new pair.

    Compared with :class:`_AdjecencyMatrixGraphStorage` this trades
    constant-time random access for sparse-graph memory savings.
    """

    _n: int
    _edge_monad: type[EM]
    _vertices: set[VertexID]
    _matrix: list[list[tuple[VertexID, EM]]]

    def __init__(self, n: int, edge_monad: type[EM]) -> None:
        """Allocate the active-vertex set and an empty adjacency list.

        :param n: Number of vertex slots to reserve.  Identifiers ``0``
            through ``n - 1`` are all marked active.
        :param edge_monad: The :class:`EdgeMonad` class used to compose
            edge identifiers between every vertex pair.
        """
        self._edge_monad = edge_monad
        self._n = n
        self._vertices = set(map(VertexID, range(n)))
        self._matrix = [[] for _ in range(n)]

    def update_edge(self, u: VertexID, v: VertexID, value: EdgeID) -> EM:
        """Insert or update the edge monad for the ``(u, v)`` pair.

        Scans ``u``'s adjacency list for an existing entry with
        destination ``v``.  If found, *value* is chained onto its monad
        in place; otherwise a fresh monad is bound from *value* and
        appended.

        :param u: Source vertex identifier.
        :param v: Destination vertex identifier.
        :param value: The edge identifier to store for ``(u, v)``.
        :return: The edge monad now associated with ``(u, v)``.
        """
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
        """Return a snapshot of the currently active vertex set.

        :return: A :class:`frozenset` of active :class:`VertexID` values.
        """
        return frozenset(self._vertices)

    def _edges(self) -> Iterable[tuple[VertexID, VertexID, EM]]:
        for u, u_v_edges in enumerate(self._matrix):
            if u not in self._vertices:
                continue
            yield from ((u, v, value) for v, value in u_v_edges)

    def _incoming_edges(self, v: VertexID) -> Iterable[tuple[VertexID, VertexID, EM]]:
        yield from ((u, w, value) for u, w, value in self._edges() if v == w)

    def _outgoing_edges(self, u: VertexID) -> Iterable[tuple[VertexID, VertexID, EM]]:
        yield from ((u, v, val) for v, val in self._matrix[u])


AdjecencyMatrixGraphStorage: type[GraphStorage] = _AdjecencyMatrixGraphStorage
LinkedListGraphStorage: type[GraphStorage] = _LinkedListGraphStorage
