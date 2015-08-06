from . import core, utils
import sys

# ================================================================
# Printing 
# ================================================================

def print_tree(outputs, o=sys.stdout, nodefn=None):
    """
    Print out a representation of the computation graph as a tree
    nodefn is called after printing the result for every node, as
    nodefn(node, o)
    So you can print more attributes of the node
    """
    if isinstance(outputs, core.Node):
        outputs = [outputs]
    node2name = {}
    expands = []
    for node in outputs:
        _print_tree(node, 0, node2name, expands, o, nodefn)
    assert expands == []
    return node2name

def _print_tree(node, depth, node2name, expands, o, nodefn):
    o.write("| "*depth)
    if node in node2name: 
        varname = node2name[node]
        new = False
    else:
        varname = node.get_name() + "@%i"%len(node2name)
        node2name[node] = varname
        new = True

    color = utils.Color.GREEN if node.is_input() else utils.Color.RED
    utils.colorprint(color, varname, o)

    if new:
        if nodefn is not None: nodefn(node, o)
        o.write("\n")
        for p in node.parents:
            _print_tree(p, depth+1, node2name, expands, o, nodefn)
    else:
        if not node.is_input(): o.write(" (see above)")
        o.write("\n")

def print_expr(x, o=sys.stdout):
    """
    Returns a string that represents a computation graph
    """
    node2s = {}
    o.write(_get_expr(x, node2s))
    o.write("\n")

def _get_expr(node, node2s):
    if node in node2s:
        return node2s[node]
    else:
        if node.is_input():
            name = node2s[node] = node.get_name() or "@%i"%len(node2s)
            return name
        else:
            parent_exprs = [_get_expr(parent, node2s) 
                for parent in node.parents]
            return node.op.get_expr(parent_exprs)

def print_text(outputs, o=sys.stdout):
    """
    Print computation graph in single-statement assignment form,
    inspired by LLVM IR. (needs work)
    """
    if isinstance(outputs, core.Node):
        outputs = [outputs]
    node2name = {}
    for node in core.topsorted(outputs):
        thisname = node2name[node] = (node.get_name() if node.is_input() else "") + "@%i"%len(node2name)
        if node.is_input():
            o.write("%s <- input\n"%thisname)
        else:
            o.write("%s = %s %s\n"%(thisname, node.op.get_name(), " ".join(node2name[parent]
                for parent in node.parents)))

def as_dot(nodes):
    if isinstance(nodes, core.Node):
        nodes = [nodes]
    from graphviz import Digraph
    g = Digraph()
    for n in core.topsorted(nodes):
        g.node(str(id(n)), str(n))
        for p in n.parents:
            g.edge(str(id(n)), str(id(p)))
    return g

