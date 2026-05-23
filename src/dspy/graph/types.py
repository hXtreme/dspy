"""Core type definitions and protocols for the graph package.

Defines the fundamental building blocks used across all graph implementations:
vertex and edge identifiers, the :class:`EdgeMonad` protocol for composing
parallel edges, the :class:`GraphStorage` protocol for backend storage, and the
high-level :class:`Graph` protocol that concrete graph classes implement.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

import textwrap
import typing
from collections.abc import Callable, Iterable, Iterator
from typing import NewType, Protocol, Self

VertexID = NewType("VertexID", int)
EdgeID = NewType("EdgeID", int)

Vertex = NewType("Vertex", int)
Edge = NewType("Edge", tuple[EdgeID, Vertex, Vertex])


class EdgeMonad(Protocol):
    """Protocol for composing multiple edge identifiers between a vertex pair.

    An edge monad accumulates :class:`EdgeID` values that share the same
    ``(source, destination)`` pair.  Implementations decide whether to keep
    all identifiers (multi-graph) or only the latest (simple graph).

    The monadic interface follows a *bind / chain / unwrap* pattern:

    * :meth:`bind` creates a fresh monad from a single edge.
    * :meth:`chain` adds another edge to an existing monad.
    * :meth:`unwrap` extracts the accumulated edge identifiers.
    """

    def __init__(self, e: EdgeID | Self) -> None:
        """Initialise the monad from an edge identifier or an existing monad.

        Implementations decide whether *e* is wrapped into a new
        single-element container or, when *e* is already an
        :class:`EdgeMonad` of the same type, copied/adopted to seed the
        new instance.

        :param e: Either a single :class:`EdgeID` to wrap, or an existing
            :class:`EdgeMonad` whose contents seed the new monad.
        """
        ...

    @classmethod
    def bind(cls, e: EdgeID) -> Self:
        """Create a new monad containing a single edge identifier.

        :param e: The initial edge identifier.
        :return: A new :class:`EdgeMonad` instance wrapping *e*.
        """
        return cls(e)

    def chain(self, e: EdgeID) -> Self:
        """Append an additional edge identifier to this monad.

        :param e: The edge identifier to add.
        :return: ``self``, updated in place, to allow fluent chaining.
        """
        ...

    def unwrap(self) -> Iterable[EdgeID]:
        """Return all accumulated edge identifiers.

        :return: An iterable of every :class:`EdgeID` held by this monad.
        """
        ...

    @classmethod
    def bind_or_chain(cls, e: EdgeID, m: Self | None) -> Self:
        """Bind *e* into a new monad if *m* is ``None``, otherwise chain onto *m*.

        Convenience factory that eliminates the ``if m is None`` branch at
        every call-site.

        :param e: The edge identifier to incorporate.
        :param m: An existing monad to extend, or ``None`` to start fresh.
        :return: A monad containing *e* (and any prior edges in *m*).
        """
        if m is None:
            return cls.bind(e)
        return m.chain(e)

    def __iter__(self) -> Iterator[int]:
        """Iterate over the raw integer values of all accumulated edge IDs.

        :return: An iterator yielding each edge identifier as an ``int``.
        """
        return iter(self.unwrap())


class GraphStorage[EM: EdgeMonad](Protocol):
    """Protocol defining a low-level, vertex-ID-based storage backend for graphs.

    A storage backend manages an ``n``-vertex graph using integer
    :class:`VertexID` indices and delegates edge composition to an
    :class:`EdgeMonad` of type *EM*.  Concrete backends (adjacency matrix,
    adjacency list, etc.) implement the private ``_edges``,
    ``_incoming_edges``, and ``_outgoing_edges`` methods; the public
    ``edges``, ``incoming_edges``, and ``outgoing_edges`` methods flatten
    the monad results into individual ``(source, destination, EdgeID)``
    triples automatically.
    """

    _edge_monad: type[EM]

    def update_edge(self, u: VertexID, v: VertexID, value: EdgeID) -> EM:
        """Insert or update an edge from vertex *u* to vertex *v*.

        If an edge monad already exists for the ``(u, v)`` pair the new
        *value* is chained onto it; otherwise a fresh monad is created.

        :param u: Source vertex identifier.
        :param v: Destination vertex identifier.
        :param value: The edge identifier to store.
        :return: The updated (or newly created) edge monad for ``(u, v)``.
        """
        ...

    def vertices(self) -> Iterable[VertexID]:
        """Return all vertex identifiers currently in the storage.

        :return: An iterable of :class:`VertexID` values.
        """
        ...

    def _edges(self) -> Iterable[tuple[VertexID, VertexID, EM]]: ...

    def _incoming_edges(
        self,
        v: VertexID,
    ) -> Iterable[tuple[VertexID, VertexID, EM]]: ...
    def _outgoing_edges(
        self,
        u: VertexID,
    ) -> Iterable[tuple[VertexID, VertexID, EM]]: ...

    def edges(self) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        """Return all edges as flattened ``(source, destination, EdgeID)`` triples.

        Multi-edges between the same vertex pair are expanded into separate
        triples via the underlying :class:`EdgeMonad`.

        :return: An iterable of ``(source, destination, edge_id)`` tuples.
        """
        return self.__flatten_edges(self._edges())

    def incoming_edges(
        self,
        v: VertexID,
    ) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        """Return all edges arriving at vertex *v* as flattened triples.

        :param v: The destination vertex identifier.
        :return: An iterable of ``(source, v, edge_id)`` tuples.
        """
        return self.__flatten_edges(self._incoming_edges(v))

    def outgoing_edges(
        self,
        u: VertexID,
    ) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        """Return all edges departing from vertex *u* as flattened triples.

        :param u: The source vertex identifier.
        :return: An iterable of ``(u, destination, edge_id)`` tuples.
        """
        return self.__flatten_edges(self._outgoing_edges(u))

    def _update_edge(self, e: EdgeID, em: EM | None) -> EM:
        return self._edge_monad.bind_or_chain(e, em)

    @staticmethod
    def __flatten_edges(
        edge_monads: Iterable[tuple[VertexID, VertexID, EM]],
    ) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        for u, v, edge_moad in edge_monads:
            yield from ((u, v, edge) for edge in edge_moad.unwrap())

    @staticmethod
    def __flaten_edge(
        u: VertexID,
        v: VertexID,
        em: EM,
    ) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        return ((u, v, e) for e in em.unwrap())

    def __str__(self) -> str:
        """Return the string reperesentation of the graph."""
        graph_id = id(self)
        vertices = set(self.vertices())
        edges = textwrap.indent(
            "\n".join([f"{u} --> {v}: {em!s}" for u, v, em in self._edges()]),
            "        ",
        )

        return textwrap.dedent(
            """
            GraphStorage (id={graph_id})
                Vertices: {vertices}
                Edges:
            {edges}
            """
        ).format(graph_id=graph_id, vertices=vertices, edges=edges)


type GraphStorageProvider[EM: EdgeMonad] = Callable[[int, type[EM]], GraphStorage[EM]]

type PropertyConstructor[T] = Callable[[], T]


class Graph[EM: EdgeMonad, VertexProp = None, EdgeProp = None](Protocol):
    """High-level protocol that all concrete graph implementations must satisfy.

    A :class:`Graph` is parameterised by:

    * *EM* -- the :class:`EdgeMonad` type used by its storage backend.
    * *VertexProp* -- the type of per-vertex property objects (default ``None``).
    * *EdgeProp* -- the type of per-edge property objects (default ``None``).

    Implementations provide vertex/edge enumeration, directed connectivity
    queries, and property access.
    """

    # Init
    def __init__(
        self,
        storage: GraphStorageProvider[EM],
        vertices: Iterable[Vertex] | int = 0,
        edges: Iterable[Edge] | None = None,
        vertex_property: PropertyConstructor[VertexProp] | None = None,
        edge_property: PropertyConstructor[EdgeProp] | None = None,
    ) -> None:
        """Construct a new graph.

        :param storage: A callable that creates a :class:`GraphStorage` backend
            given a vertex count and an edge-monad type.
        :param vertices: Either an integer count (vertices are auto-generated as
            ``Vertex(VertexID(0))`` through ``Vertex(VertexID(n-1))``), or an
            explicit iterable of :class:`Vertex` objects.
        :param edges: An optional iterable of :class:`Edge` tuples to insert
            during construction.  Each edge is a ``(EdgeID, source, destination)``
            triple.
        :param vertex_property: A zero-argument factory called once per vertex to
            produce its initial property value.  ``None`` means properties default
            to ``None``.
        :param edge_property: A zero-argument factory called once per edge to
            produce its initial property value.  ``None`` means properties default
            to ``None``.
        """
        ...

    # Building blocks
    def vertices(self) -> Iterable[Vertex]:
        """Return all vertices in the graph.

        :return: An iterable of :class:`Vertex` objects.
        """
        ...

    def edges(self, vertex: Vertex | None = None) -> Iterable[Edge]:
        """Return edges in the graph.

        :param vertex: If provided, return only the outgoing edges of this
            vertex.  If ``None``, return all edges in the graph.
        :return: An iterable of :class:`Edge` tuples ``(EdgeID, source, dest)``.
        """
        ...

    # Connectivity
    def incoming_edges(self, vertex: Vertex) -> Iterable[Edge]:
        """Return all edges whose destination is *vertex*.

        :param vertex: The target vertex.
        :return: An iterable of :class:`Edge` tuples arriving at *vertex*.
        """
        ...

    def outgoing_edges(self, vertex: Vertex) -> Iterable[Edge]:
        """Return all edges whose source is *vertex*.

        :param vertex: The source vertex.
        :return: An iterable of :class:`Edge` tuples departing from *vertex*.
        """
        ...

    # Meta-information
    @typing.overload
    def property(self, item: Vertex) -> VertexProp: ...
    @typing.overload
    def property(self, item: Edge) -> EdgeProp: ...

    # Magic Methods
    def __str__(self) -> str:
        """Return the string reperesentation of the graph."""
        ...
