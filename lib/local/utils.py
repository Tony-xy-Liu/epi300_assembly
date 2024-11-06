from __future__ import annotations
import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal
from scipy.cluster.hierarchy import to_tree, linkage, ClusterNode, dendrogram
from scipy.spatial.distance import squareform

def safe_log10(x):
    if isinstance(x, np.ndarray):
        negs = np.ones_like(x)
        negs[x<0] = -1
        return  negs * np.log10(negs*x+1)
    else:
        x = np.log10(x+1) if x>0 else -np.log10(1-x)
        return x

def pd_set_type(cols: list|str, t, df: pd.DataFrame):
    if isinstance(cols, str): cols = cols.split(', ')
    for col in cols: df[col] = df[col].astype(t)

def regex(r, s):
    for m in re.finditer(r, s):
        yield s[m.start():m.end()]

# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def add_to_python_path(paths: list[Path|str]):
    if not isinstance(paths, list): paths = [paths]
    sys.path = list(set(sys.path + [str(p) for p in paths]))

@dataclass
class DendrogramNode[T]:
    x: float
    y: float
    i: T
    left: DendrogramNode[T]|None
    right: DendrogramNode[T]|None
    def IsLeaf(self):
        return self.left is None and self.right is None
    
    def Children(self):
        for c in [self.left, self.right]:
            if c is not None:
                yield c

    def Traverse(self, order="in"):
        if self.IsLeaf():
            yield self
            return
        
        left, right = self.Children()
        todo = {
            "pre": [self, left, right],
            "in": [left, self, right],
            "post": [left, right, self]
        }[order]
        for x in todo:
            if x == self:
                yield self
            else:
                yield from x.Traverse(order)
    
@dataclass
class LinkageResult[T]:
    labels: list[T]
    mat: np.ndarray
    tree: DendrogramNode
    linkage: Any

def HierarchicalCluster(Z: np.ndarray, labels: list|None = None, method="ward", metric="euclidean", distance_sort=False, count_sort=False) -> LinkageResult:
    class FloatDict[U]:
        def __init__(self, d: dict[float, U] = dict()):
            self.d = d
            self.index = sorted(list(d.keys()))

        def _binary_search(self, k: float):
            # binary search for the nearest key to account for floating point errors
            l, r = 0, len(self.index)-1
            while l < r:
                m = (l + r) // 2
                if self.index[m] < k:
                    l = m + 1
                else:
                    r = m
            return self.index[l]

        def Nearest(self, k: float):
            if k in self.d: return self.d[k]
            return self.d[self._binary_search(k)]

    @dataclass
    class _DendNode:
        x: float
        y: float
        i: int
        xs: tuple[float, float, float, float]
        ys: tuple[float, float, float, float]
        left: _DendNode|None
        right: _DendNode|None

        def __hash__(self) -> int:
            return self.i
        def IsLeaf(self):
            return self.left is None and self.right is None
        def LeftCoords(self):
            (x1, x2, x3, x4), (y1, y2, y3, y4) = self.xs, self.ys
            return x1, y1
        def RightCoords(self):
            (x1, x2, x3, x4), (y1, y2, y3, y4) = self.xs, self.ys
            return x4, y4
        def IsAt(self, x, y):
            if x == self.x and y == self.y: return True
            zero = 1.0e-4
            return abs(self.x - x) < zero and abs(self.y - y) < zero
        
    # ###############################################################
    # use scipy to do the clustering
    labels = list(labels) if labels is not None else list(range(Z.shape[0]))
    _Z = Z
    if metric=="precomputed": _Z = squareform(Z)
    linkage_data = linkage(_Z, method=method, metric=metric, optimal_ordering=True)
    _p: Any = dict(count_sort=count_sort, distance_sort=distance_sort) # to bypass type warning
    dend = dendrogram(linkage_data, no_plot=True, labels=labels, **_p)

    # ###############################################################
    # convert the arrays of inscrutible numbers and unhelpful tree
    # tree structure from scipy into something useful 
    link_ys = dend["dcoord"]
    link_xs = dend["icoord"]
    ordered_labels = dend["ivl"]

    _nodes_by_x = {}
    _nodes_by_y = {}
    nodes_by_i: dict[int, _DendNode] = {}
    nodes: list[_DendNode] = []
    for i, (xs, ys) in enumerate(zip(link_xs, link_ys)):
        (x1, x2, x3, x4), (y1, y2, y3, y4) = xs, ys
        nx = (x2+x3)/2
        ny = y2
        node = _DendNode(nx, ny, i, xs, ys, None, None)
        _nodes_by_x[nx] = _nodes_by_x.get(nx, []) + [i]
        _nodes_by_y[ny] = _nodes_by_y.get(ny, []) + [i]
        nodes_by_i[i] = node
        nodes.append(node)
    nodes_by_x = FloatDict(_nodes_by_x)
    nodes_by_y = FloatDict(_nodes_by_y)

    def get_node_at(x, y):
        candidates = set(nodes_by_x.Nearest(x) + nodes_by_y.Nearest(y))
        candidates = [i for i in candidates if nodes_by_i[i].IsAt(x, y)]
        if len(candidates) == 0: return
        assert len(candidates) == 1
        return nodes_by_i[candidates[0]]

    node_parents: dict[_DendNode, _DendNode] = {}
    for n in nodes:
        for k, (x, y) in [("left", n.LeftCoords()), ("right", n.RightCoords())]:
            child = get_node_at(x, y)
            if child is None: continue
            setattr(n, k, child)
            node_parents[child] = n

    _root = None
    for n in nodes:
        if n in node_parents: continue
        assert _root is None
        _root = n
    assert _root is not None

    root = DendrogramNode(_root.x, _root.y, _root.i, None, None)
    todo: list[DendrogramNode] = [root]
    label_index = 0
    # seen = set()
    while len(todo) > 0:
        new = todo.pop()
        # if new.i in seen: continue
        # seen.add(new.i)
        if new.i == -1:
            new.i = ordered_labels[label_index]
            label_index += 1
            continue
        _node = nodes_by_i[new.i]

        _ch_coords = [_node.LeftCoords(), _node.RightCoords()]
        _chl, _chr = sorted(_ch_coords, key=lambda x: x[0], reverse=True) # by x        
        for ch, (chx, chy) in [("right", _chl), ("left", _chr)]:
            _child = getattr(_node, ch)
            # if _child is None or _child.i in seen:
            if _child is None:
                new_child = DendrogramNode(chx, chy, -1, None, None)
            else:
                new_child = DendrogramNode(chx, chy, _child.i, None, None)
            setattr(new, ch, new_child)
            todo.append(new_child)

    max_x, min_x, = -np.inf, np.inf
    for node in root.Traverse():
        max_x = max(max_x, node.x)
        min_x = min(min_x, node.x)
    for node in root.Traverse():
        node.x = (node.x - min_x) / (max_x - min_x)
    
    # ###############################################################
    # order the matrix based on the clustering

    new_order = [labels.index(l) for l in ordered_labels]
    # mat = np.zeros(shape=Z.shape, dtype=Z.dtype)
    # for j, i in enumerate(new_order):
    #     mat[j, :] = Z[i, :]

    return LinkageResult(ordered_labels, Z[new_order], root, linkage_data)
