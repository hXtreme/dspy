import typing
from collections.abc import Callable, Iterable, Iterator
from typing import NewType, Protocol, Self

VertexID = NewType("VertexID", int)
EdgeID = NewType("EdgeID", int)

Vertex = NewType("Vertex", VertexID)
Edge = NewType("Edge", tuple[EdgeID, Vertex, Vertex])

class EdgeMonad(Protocol):
    @classmethod
    def bind(cls, e: EdgeID) -> Self: ...
    def chain(self, e: EdgeID) -> Self: ...
    def unwrap(self) -> Iterable[EdgeID]: ...

    @classmethod
    def bind_or_chain(cls, e: EdgeID, m: Self | None) -> Self:
        if m is None:
            return cls.bind(e)
        return m.chain(e)
    def __iter__(self) -> Iterator[int]:
        return iter(self.unwrap())

class GraphStorage[EM: EdgeMonad](Protocol):
    _edge_monad: type[EM]

    def update_edge(self, u: VertexID, v: VertexID, value: EdgeID) -> EM: ...
    def vertices(self) -> Iterable[VertexID]: ...
    def _edges(self) -> Iterable[tuple[VertexID, VertexID, EM]]: ...

    def _incoming_edges(
        self, v: VertexID,
    ) -> Iterable[tuple[VertexID, VertexID, EM]]: ...
    def _outgoing_edges(
        self, u: VertexID,
    ) -> Iterable[tuple[VertexID, VertexID, EM]]: ...

    def edges(self) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        return self.__flatten_edges(self._edges())
    def incoming_edges(
        self, v: VertexID,
    ) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        return self.__flatten_edges(self._incoming_edges(v))
    def outgoing_edges(
        self, u: VertexID,
    ) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        return self.__flatten_edges(self._outgoing_edges(u))

    def _update_edge(self, e: EdgeID, em: EM | None) -> EM:
        return self._edge_monad.bind_or_chain(e, em)
    @staticmethod
    def __flatten_edges(
        edge_monads: Iterable[tuple[VertexID, VertexID, EM]],
    ) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        for (u, v, edge_moad) in edge_monads:
            yield from ((u, v, edge) for edge in edge_moad.unwrap())
    @staticmethod
    def __flaten_edge(
        u: VertexID, v: VertexID, em: EM,
    ) -> Iterable[tuple[VertexID, VertexID, EdgeID]]:
        return ((u, v, e) for e in em.unwrap())

type GraphStorageProvider[EM: EdgeMonad] = Callable[[int, type[EM]], GraphStorage[EM]]

type PropertyConstructor[T] = Callable[[], T]

class Graph[EM: EdgeMonad, VertexProp = None, EdgeProp = None](Protocol):
    # Init
    def __init__(
        self,
        storage: GraphStorageProvider[EM],
        vertices: Iterable[Vertex] | int = 0,
        edges: Iterable[Edge] | None = None,
        vertex_property: PropertyConstructor[VertexProp] | None = None,
        edge_property: PropertyConstructor[EdgeProp] | None = None,
    ) -> None: ...

    # Building blocks
    def vertices(self) -> Iterable[Vertex]: ...
    def edges(self, vertex: Vertex | None = None) -> Iterable[Edge]: ...

    # Connectivity
    def incoming_edges(self, vertex: Vertex) -> Iterable[Edge]: ...
    def outgoing_edges(self, vertex: Vertex) -> Iterable[Edge]: ...

    # Meta-information
    @typing.overload
    def property(self, item: Vertex) -> VertexProp: ...
    @typing.overload
    def property(self, item: Edge) -> EdgeProp: ...
