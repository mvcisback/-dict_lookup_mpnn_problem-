from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np


__all__ = ['ProblemFactory', 'gen_problems']


@dataclass
class Problem:
    n_keys: int
    nodes: np.array
    adj: np.array
    answers: np.array

    @property
    def n_vals(self) -> int:
        return self.nodes.shape[1] - self.n_keys

    def decode(self, x: np.array) -> tuple[int, int | None]:
        n = self.n_keys
        key = int(x[:n].argmax())
        val = int(x[n:].argmax())
        val = None if x[n + val] == 0 else val
        return (key, val)


@dataclass
class ProblemFactory:
    n_keys: int
    n_vals: int
    rng: np.random.Generator

    def encode(self, key: int, val: int | None = None) -> np.array:
        data = np.zeros(self.n_keys + self.n_vals)
        data[key] = 1
        if val is not None:
            data[self.n_keys + val] = 1
        return data

    def gen_problem(self) -> Problem:
        n = self.rng.integers(1, self.n_keys, endpoint=True)
        keys = np.arange(n)
        self.rng.shuffle(keys)
        vals = self.rng.choice(self.n_vals, n)

        # Bipartite adjacency matrix is block anti-diagonal.
        # Further, lookup assumes full connectivity between colors.
        anti_diag = np.fliplr(np.eye(2))
        adj_mat = np.kron(anti_diag, np.ones((n, n))) 

        # Each node has encoding of (key, None) or (key, val).
        key_feats = [self.encode(k) for k in keys]
        key_val_feats = [self.encode(k, v) for k, v in zip(keys, vals)]
        node_feats = np.concat([key_feats, key_val_feats])
        return Problem(n_keys=self.n_keys, nodes=node_feats,
                       adj=adj_mat, answers=vals)


def gen_problems(n_keys: int, n_vals: int, seed: int = 0) -> Iterator[Problem]:
    rng = np.random.default_rng(seed=seed)
    factory = ProblemFactory(n_keys, n_vals, rng)
    while True:
        yield factory.gen_problem()
