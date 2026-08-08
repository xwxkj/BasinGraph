# Track C implementation note 002: portable source identity

Status: **registered before any Track C objective evaluation**  
Date: 2026-08-08  
Track C confirmatory objective evaluations before correction: **0**

The first materialized identity inadvertently listed interpreter-generated `__pycache__` and `.pyc` files. These files are not source artifacts and are Python-version and platform dependent. They have been removed from the repository and permanently excluded from the Track C source-identity traversal. No algorithm, task, parameter, dataset, reference value, budget or endpoint changed.
