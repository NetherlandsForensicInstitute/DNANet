# Data Utilities

The `dnanet.data.utils` module provides small, stateless helper functions
that are dataset-agnostic and kit-agnostic.

## Functions

```python
from dnanet.data.utils import generate_random_name, find_files_by_suffix
```

- `generate_random_name()` — Generate a random human-readable experiment name
  (e.g., "BrilliantFalcon") using the `coolname` library.
- `find_files_by_suffix(root, suffix)` — Recursively find all files with a
  given suffix under a root directory.
