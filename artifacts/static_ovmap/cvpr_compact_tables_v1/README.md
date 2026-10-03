# Compact measured evidence

The direct tables, metrics, PDF, freeze and validation summaries accompany `evidence.tar.xz`. The archive contains the original byte-for-byte JSON scores, decisions, matching traces, class metrics, costs, failures and original cold receipts. `bundle.json` lists every member's original SHA256 and size.

From this directory, restore detailed evidence into a new directory with standard tools:

```bash
mkdir compact_evidence
tar -xJf evidence.tar.xz -C compact_evidence
```

`compact_evidence/shared_artifacts.json` records immutable large dependencies and the production reconstruction command. Relocate roots with `--path-map OLD=NEW`. Original reserved cold observations must be restored, never replayed. Archive extraction performs no inference.
