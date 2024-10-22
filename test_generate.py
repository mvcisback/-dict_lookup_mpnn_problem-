import numpy as np
import networkx as nx
from bidict import bidict
from networkx.algorithms import bipartite as B

from dict_lookup_mpnn_problem import gen_problems


def test_smoke():
    problems = gen_problems(n_keys=2, n_vals=3)

    for i, prob in zip(range(3), problems):
        p = next(problems)
        graph = nx.Graph(p.adj)

        assert B.is_bipartite(graph)
        assert p.nodes.shape[0] == 2 * len(p.answers)
        assert len(p.nodes) == p.adj.shape[0] == p.adj.shape[1]

        n = len(p.answers)
        keys = set()
        for node in p.nodes[:n]:
            key, val = p.decode(node)
            keys.add(key)
            assert val == None
        assert len(keys) == n

        f = bidict()
        for answer, node in zip(p.answers, p.nodes[n:]):
            key, val = p.decode(node)
            f[key] = val
            assert answer == val
        assert keys == set(f.keys())