"""Rewrites an SMPL .pkl without chumpy objects, so smplx can load it on numpy 2.

The published SMPL pickles hold chumpy arrays and reference a pre-1.8 scipy
sparse path, so unpickling them needs both packages at their old versions. This
resolves those references to plain numpy/scipy objects and writes the result
back out, leaving the environment untouched.
"""

import pickle
import sys

import numpy as np
import scipy.sparse


class _Ch:
    """Stands in for chumpy.Ch: keeps whatever state the pickle carries."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Ch
        # scipy.sparse.csc -> scipy.sparse._csc since scipy 1.8
        if module.startswith("scipy.sparse.") and not module.startswith("scipy.sparse._"):
            return getattr(scipy.sparse, name)
        return super().find_class(module, name)


def to_plain(obj):
    """Converts chumpy stand-ins to numpy, recursing through containers."""
    if isinstance(obj, _Ch):
        for key in ("x", "r", "_data"):
            if key in obj.__dict__:
                return np.asarray(obj.__dict__[key])
        raise ValueError(f"chumpy object without a value: {list(obj.__dict__)}")
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_plain(v) for v in obj)
    return obj


def main(src, dst):
    with open(src, "rb") as f:
        data = _Unpickler(f, encoding="latin1").load()

    cleaned = to_plain(data)

    n_ch = sum(isinstance(v, _Ch) for v in data.values())
    print(f"keys: {len(cleaned)} | chumpy values converted: {n_ch}")
    for k, v in sorted(cleaned.items()):
        kind = type(v).__name__
        shape = getattr(v, "shape", "")
        print(f"   {k:<28}{kind:<14}{shape}")

    with open(dst, "wb") as f:
        pickle.dump(cleaned, f, protocol=4)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
