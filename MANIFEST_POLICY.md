# Repository manifest policy

The root manifest present in the original public `main` branch was generated
before the result-bearing `basingraph_v2/` implementation and later prospective
evidence were added. It therefore described the historical simplified
`basingraph/optimizer.py` snapshot and was not a valid manifest for the current
repository tree.

Step 0 removes that stale file rather than presenting incorrect checksums.
A current root manifest must be generated after checkout and before a release:

```bash
python scripts_v2/generate_repository_manifest.py
python scripts_v2/audit_step0_identity.py --require-manifest
```

The generator prefers `git ls-files`, excludes the manifest itself to avoid
self-reference, and writes both:

- `MANIFEST_SHA256.csv`;
- `MANIFEST_SHA256.scope.json`.

Every B21 output directory is separately sealed with its own SHA-256 manifest,
so repository-source integrity and experimental-output integrity remain
independent.

Historical manifests remain available through earlier Git commits and archived
Zenodo artifacts. They must not be relabelled as current manifests.
