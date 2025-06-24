# Visualization code for autograd graphs using Graphviz
# Much of this code is inspired by Andrej Karpathy's micrograd visualization code
from graphviz import Digraph

def _trace(root):
    nodes, edges = set(), []

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.append((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="png", graph_attr={"rankdir": "LR"})
    nodes, edges = _trace(root)

    for n in nodes:
        uid = str(id(n))
        label = f"{n.data}"
        if n.operation:
            label += f" | {n.operation}"
        label += f" | ∇={n.grad:.3g}"
        dot.node(uid, label=label, shape="record")

    for src, dst in edges:
        dot.edge(str(id(src)), str(id(dst)))

    return dot