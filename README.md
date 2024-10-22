This project implements a graph generator inspired by the DictionaryLookup Benchmark from
["How Attentive are Graph Neural Networks"](https://github.com/tech-srl/how_attentive_are_gats/tree/main/dictionary_lookup).

![Problem Image](https://github.com/tech-srl/how_attentive_are_gats/raw/main/dictionary_lookup/images/fig2.png)

The problem is described as follows:

1. A bi-partite graph, `G`, is provided, with nodes split into two sets `TOP`, `BOT`.

2. The set `TOP` is a subset of `Keys`.

3. The set `BOT` is a bijection `Keys -> Values` represented as a subset of
   `Keys x Values`.

  - For example if `("A", 1)` is in `BOT` then no other element can have
    `"A"` for a key or `1` as a value.

4. The goal is to map each key in TOP to the approriate value via `BOT(key)`.

The encoding provided by this repo is to take: `Key = 1..n_keys` and `Values = 1..n_vals`
for some integers, `n_vals` and `n_vals`. The primary interface to this code is the
`gen_problems` function which takes these in along with a random seed and provides
an infinite sequence of dictionary lookup problems.

```python
from dict_lookup_mpnn_problem import gen_problems

problems = gen_problems(n_keys=2, n_vals=3, seed=12)
p1 = next(problems)
```

- Here `p1` is a `Problem` object with three relevant attributes:
   1. `adj`: Adjacency matrix (it's block anti-diagonal).
   1. `nodes`: Array of node features (see below).
   1. `answers`: What value the `i`th node should decode to.
- The feature encoding is the 1-hot encoding of the key concatenated with the 1-hot
  encoding of the value. If no value is present, the all zeros vector is used.

An example output for `n_keys=2` and `n_vals=3` is provided below.

```python
Problem(nodes=array([[1., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [1., 0., 1., 0., 0.],
                     [0., 1., 0., 0., 1.]]),
        adj=array([[0., 0., 1., 1.], 
                   [0., 0., 1., 1.],
                   [1., 1., 0., 0.],
                   [1., 1., 0., 0.]]),
        answers=array([0, 2]),
        n_keys=2)
```
