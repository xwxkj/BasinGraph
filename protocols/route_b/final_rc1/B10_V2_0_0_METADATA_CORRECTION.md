# BasinGraph v2.0.0 metadata correction

Software version DOI: 10.5281/zenodo.20789002
Reproducibility dataset DOI: 10.5281/zenodo.20788903

The original v2.0.0 tag contains a duplicated DOI prefix in `.zenodo.json`
and `RELEASE_NOTES_v2.0.0.md`. The authoritative dataset DOI is
10.5281/zenodo.20788903.

This was a metadata-only error. It did not alter:

- BasinGraph source code;
- the selected 2.0.0-rc1 implementation;
- the frozen options hash;
- COCO/BBOB or CUTEst benchmark evidence;
- any reported numerical result.

The published software record metadata and the current repository branch were
corrected without moving or recreating the immutable v2.0.0 tag.
