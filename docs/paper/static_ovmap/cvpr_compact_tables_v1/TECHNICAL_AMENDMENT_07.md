# Technical amendment 07: chronological publication audit

Amendment06 produced an actual verified release. Its54 physical files and
manifest occupy42,740,397 bytes, leaving9,688,403 bytes before the primary review.
Every original archived member and1,658 copied source file was restored and
checked by original byte count and SHA256. Scientific coverage remains168/172,
10/14 and22/24; original scientific/cold artifacts and the inspected PDF are exact.

The primary read all327 catalogued requirements and found a publication ordering
defect: the prepublication gate required every requirement to be already proven,
including the normal push, remote SHA verification, external post-push receipt
and final response. These actions necessarily follow the prepublication review.
Scientific blocks cannot serve as substitute evidence for these later actions.

The prepublication review now allows PENDING_PUBLICATION only for six explicitly
identified original requirements, with their exact subsequent action named.
They remain in unresolved_required_items and pending_publication_steps. The
status is READY_FOR_PUBLICATION[_WITH_TECHNICAL_BLOCKS], objective_complete=false.
No scientific, implementation, missing-data or generic uncertain requirement can
use this state. All327 requirements and all physical/archived files still require
primary findings and evidence. Actual10 scientific dependency tokens remain
fully covered and unresolved. The final external push receipt and postpublication
review must resolve the later obligations from actual execution; readiness never
proves upload, full scientific completion or objective completion.

Only publication.py and the existing review-gate assertions change. The reports
remain byte-identical with their original439128cd generation revision; the
publisher already verifies their generator at that committed ancestor. All15
scientific producers, numeric artifacts, diagnostics, tables/PDF and22 original
cold calls remain unchanged. No inference or scientific producer is replayed.
