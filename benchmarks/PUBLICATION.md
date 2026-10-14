# Published benchmark copies

Machine-specific storage prefixes in the retained benchmark archive are normalized
as `/workspace`. Numerical readings, failed outcomes, source and producer identities,
protocols and declared acceptance gates are preserved. This normalization is a
publication edit, not a rerun or an independent scientific confirmation.

[publication-manifest.json](publication-manifest.json) records the original and
published byte hashes and sizes for each changed file, including decompressed bytes.
Canonical receipt envelopes are re-digested after the path edit; their original
digests are recorded alongside the new ones. References in historical reviews to
original producer or receipt hashes retain that meaning. Use the manifest to map
those fingerprints to the published copies.

Archive manifests describe the published bytes and retain their original values
in `original_*` fields. Historical scripts containing normalized paths are publication
copies; their original producer hashes identify the code that ran. They are not
portable clean-clone reproduction commands. Local-only checkpoints and streams do
not become bundled artifacts because a storage path is shown.

From the repository root, check the published bytes and receipt envelopes:

```sh
python benchmarks/verify_publication.py
```

This check does not reconstruct original machine paths, authenticate that an
experiment occurred, or replay a training trajectory. Use each instrument's own
verifier for its numerical and source-admission checks.
