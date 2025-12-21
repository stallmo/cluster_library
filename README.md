# Pythonic Implementations of Cluster Algorithms

## Overview
This repository provides a compact Python implementation of fuzzy c-means clustering (fuzzy clustering). It includes the core implementation, unit tests, and a GitHub Actions workflow that runs tests on merges to `main` and on manual dispatch.

## Key files
\- `fuzzy_cmeans.py` \- Fuzzy C-Means implementation (class `FuzzyCMeans`)  
\- `.github/workflows/run-tests.yml` \- CI workflow: runs unit tests on pull requests to `main` and on manual trigger  
\- `tests/` \- (expected) unit tests for the implementation (run with `pytest`)  
\- `README.md` \- This file  
\- `requirements.txt` \- (optional) dependency pinned list used by CI

## Requirements
Tested with \- Python 3.9 \- 3.12 (CI matrix)  
\- NumPy  
\- SciPy  
\- PyTest (for running tests)

Install (example):
`python -m pip install --upgrade pip`  

`python -m pip install -r requirements.txt`

## Avoiding import errors when running tests
We plan to simplify the testing setup by providing docker containers with the necessary dependencies.

For now: If you see `ModuleNotFoundError: No module named 'fuzzy_cmeans'` when running tests or importing locally, run tests from the repository root and ensure the project root is on `PYTHONPATH`. Example (macOS / Linux):

`export PYTHONPATH=$(pwd)`  
`PYTHONPATH=$(pwd) python -m pytest -q`

Or run directly with the current path in a single command:

`PYTHONPATH=$(pwd) python -m pytest -q`

## Usage example
(From repository root; ensure dependencies installed)

    import numpy as np
    from fuzzy_cmeans import FuzzyCMeans

    X = np.array([[1.0, 2.0],
                  [1.5, 1.8],
                  [5.0, 8.0]])
    fcm = FuzzyCMeans(num_clusters=2)
    fcm.fit(X)
    U = fcm.predict(X)  # fuzzy membership matrix

## Running tests locally
`PYTHONPATH=$(pwd) python -m pytest -q`

CI uses the same command in the workflow.

## Continuous Integration
The workflow at ` .github/workflows/run-tests.yml `:
\- triggers on pull requests targeting `main` (opened/synchronize/reopened/ready_for_review)  
\- supports manual runs via `workflow_dispatch`  
\- tests on Python 3.9, 3.10, 3.11 and 3.12  
\- installs dependencies (if `requirements.txt` exists) and runs `pytest`

## Contributing
If you want to contribute, reach out to us directly on GitHub or via email.
Pull requests are welcome! Please ensure tests pass before submitting.
