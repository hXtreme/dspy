import typing
from collections.abc import Iterable
from types import NoneType
from typing import TYPE_CHECKING, Self

from dspy.graph.types import (
    Edge,
    EdgeID,
    EdgeMonad,
    Graph,
    GraphStorageProvider,
    PropertyConstructor,
    Vertex,
    VertexID,
)


class _MultiGraphEdgeMonad(EdgeMonad):
    _e: set[EdgeID]

    def __init__(self, e: EdgeID) -> None:
        self._e = {e}

    @classmethod
    def bind(cls, e: EdgeID) -> Self:
        return cls(e)
    def chain(self, e: EdgeID) -> Self:
        self._e.add(e)
        return self
    def unwrap(self) -> Iterable[EdgeID]:
        return frozenset(self._e)

def _none_property() -> None:
    return None

class AsymmetricMultiDiGraph[VertexProp = NoneType, EdgeProp = NoneType](
    Graph[_MultiGraphEdgeMonad, VertexProp, EdgeProp],
):
    __vertices: dict[Vertex, tuple[VertexID, VertexProp]]
    __edges: dict[EdgeID, EdgeProp]
    __inv_vertices: dict[VertexID, Vertex]

    def __init__(
        self,
        storage: GraphStorageProvider[_MultiGraphEdgeMonad],
        vertices: Iterable[Vertex] | int = 0,
        edges: Iterable[Edge] | None = None,
        vertex_property: PropertyConstructor[VertexProp] | None = None,
        edge_property: PropertyConstructor[EdgeProp] | None = None,
    ) -> None:
        if isinstance(vertices, int):
            vertices = (Vertex(VertexID(i)) for i in range(vertices))
        if vertex_property is None:
            if TYPE_CHECKING:
                assert VertexProp is NoneType
            vertex_property = _none_property
        if edge_property is None:
            if TYPE_CHECKING:
                assert EdgeProp is NoneType
            edge_property = _none_property

        self.__vertices = {
            v: (VertexID(v_id), vertex_property()) for v_id, v in enumerate(vertices)
        }
        self.__inv_vertices = {
            v: k for k, (v, _) in self.__vertices.items()
        }
        self.__storage = storage(len(self.__vertices), _MultiGraphEdgeMonad)
        self.__edges = {}
        for edge in edges or ():
            edge_id, u, v = edge
            self.__storage.update_edge(u, v, edge_id)
            self.__edges[edge_id] = edge_property()


    # Building blocks
    def vertices(self) -> Iterable[Vertex]:
        return self.__vertices.keys()

    def edges(self, vertex: Vertex | None = None) -> Iterable[Edge]:
        raw_edges = (
            self.__storage.outgoing_edges(vertex) if vertex is not None
            else self.__storage.edges()
        )
        yield from (
            (e, self.__inv_vertices[u], self.__inv_vertices[v]) for u, v, e in raw_edges
        )

    # Connectivity
    def incoming_edges(self, vertex: Vertex) -> Iterable[Edge]:
        raw_edges = self.__storage.incoming_edges(vertex)
        yield from (
            (e, self.__inv_vertices[u], self.__inv_vertices[v]) for u, v, e in raw_edges
        )

    def outgoing_edges(self, vertex: Vertex) -> Iterable[Edge]:
        raw_edges = self.__storage.outgoing_edges(vertex)
        yield from (
            (e, self.__inv_vertices[u], self.__inv_vertices[v]) for u, v, e in raw_edges
        )

    # Meta-information
    @typing.overload
    def property(self, item: Vertex) -> VertexProp: ...
    @typing.overload
    def property(self, item: Edge) -> EdgeProp: ...

    def property(self, item: Vertex | Edge) -> VertexProp | EdgeProp:
        if isinstance(item, tuple):
            return self.__edges[item[0]]
        if isinstance(item, int):
            return self.__vertices[item][1]
        error_msg = (
            f"Unexpected item of type {type(item)}, should be either Vertex or Edge"
        )
        raise TypeError(error_msg)

