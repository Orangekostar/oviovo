"""Causal routing by complete current segment membership, never final owners."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .query_state import ObjectQueryState, update_cached_class_scores


class CurrentLineage:
    """Keep paid ancestry; discard ambiguous split features without refunding.

    Instance numbers can be reused or reversed, so only native segment aliases
    persist. A retained feature is routed only if ALL its original member
    segments currently resolve to one positive owner. Dropped/evicted features
    never reappear, even if those owners later merge again.
    """

    def __init__(self):
        self.aliases = {}
        self.membership = {}
        self.requests = {}
        self.paid = {}
        self.geometric = []
        self.dropped_feature_ids = set()
        self.frame_diagnostics = []

    def _resolve(self, label):
        visited = set()
        while label in self.aliases:
            if label in visited:
                raise ValueError("BLOCKED_CAUSAL_LINEAGE: cyclic native segment aliases")
            visited.add(label)
            label = self.aliases[label]
        return label

    def _owners(self, segments):
        return {int(self.membership.get(self._resolve(segment), 0)) for segment in segments}

    def advance(self, snapshot, state, combine, text):
        if snapshot.get("label_instances_scope") != "all_known_labels":
            raise ValueError("BLOCKED_CAUSAL_LINEAGE: complete current native membership is missing")
        previous_membership = self.membership
        for alias in snapshot.get("aliases", []):
            old, new = int(alias["old_label"]), int(alias["resolved_label"])
            if old != new:
                self.aliases[old] = new
        rows = snapshot["label_instances"]
        self.membership = {int(row["segment_label"]): int(row["instance_label"]) for row in rows}
        if len(self.membership) != len(rows) or any(label < 0 or owner < 0 for label, owner in self.membership.items()):
            raise ValueError("BLOCKED_CAUSAL_LINEAGE: invalid complete native membership")
        for label in self.aliases:
            self._resolve(label)
        if state.aliases:
            raise ValueError("current membership cannot be mixed with permanent instance aliases")
        previous_features = {feature.request_id: feature for obj in state.objects.values() for feature in obj.features}
        objects = {}

        def obj(owner):
            return objects.setdefault(owner, ObjectQueryState())

        unavailable = set()
        for request_id, success in self.paid.items():
            row = self.requests[request_id]
            destinations = self._owners(row["segments"])
            if len(destinations) != 1 or next(iter(destinations)) <= 0:
                unavailable.update(destinations - {0})
                if request_id in previous_features:
                    self.dropped_feature_ids.add(request_id)
                continue
            owner = next(iter(destinations))
            current = obj(owner)
            current.attempted_request_ids.add(request_id)
            current.attempts += 1
            current.successes += int(success)
            current.best_paid_overlap = max(current.best_paid_overlap, row["overlap"])
            if current.last_paid_frame is None or row["frame"] >= current.last_paid_frame:
                current.last_paid_frame, current.last_paid_pose = row["frame"], row["pose"]
            if success:
                current.last_success_frame = max(current.last_success_frame or 0, row["frame"])
            if request_id in previous_features and request_id not in self.dropped_feature_ids:
                current.retain_feature(replace(previous_features[request_id], owner_id=owner))
        for segments, cells in self.geometric:
            destinations = self._owners(segments)
            if len(destinations) == 1 and next(iter(destinations)) > 0:
                obj(next(iter(destinations))).geometric_cells.update(cells)
            else:
                unavailable.update(destinations - {0})
        for owner in unavailable:
            obj(owner).lineage_available = False
        state.objects = objects
        for owner in objects:
            update_cached_class_scores(state, owner, text)

        # Preserve the native geometric coverage side effects for unambiguous
        # ancestry; a split has no well-defined inherited center/grid.
        coverage, centers = {}, {}
        for owner in sorted(combine.coverage_by_owner):
            members = {label for label, previous in previous_membership.items() if previous == owner}
            destinations = self._owners(members)
            if len(destinations) == 1 and next(iter(destinations)) > 0:
                destination = next(iter(destinations))
                coverage.setdefault(destination, set()).update(combine.coverage_by_owner[owner])
                if owner in combine.center_by_owner:
                    centers.setdefault(destination, combine.center_by_owner[owner])
        combine.coverage_by_owner, combine.center_by_owner = coverage, centers
        combine.successful_overlaps_by_owner = {}
        for request_id, success in self.paid.items():
            destinations = self._owners(self.requests[request_id]["segments"])
            if success and len(destinations) == 1 and next(iter(destinations)) > 0:
                combine.successful_overlaps_by_owner.setdefault(next(iter(destinations)), []).append(
                    self.requests[request_id]["overlap"])
        self.frame_diagnostics.append({"unavailable_history_owners": sorted(unavailable),
            "dropped_features_total": len(self.dropped_feature_ids), "known_segments": len(self.membership)})

    def register_candidates(self, candidates):
        for candidate in candidates:
            members = frozenset(self._resolve(label) for label, owner in self.membership.items()
                                if label > 0 and owner == candidate.owner_id)
            if not members or self._owners(members) != {candidate.owner_id}:
                raise ValueError("BLOCKED_CAUSAL_LINEAGE: current request has no complete positive ancestry")
            row = {"segments": members, "frame": candidate.frame_index, "overlap": candidate.overlap_pixels,
                   "pose": np.array(candidate.camera_pose, copy=True)}
            if candidate.request_id not in self.requests:
                self.requests[candidate.request_id] = row

    def record(self, candidates, results):
        for result in results:
            if result.request_id in self.paid:
                raise ValueError("exact query request was paid twice")
            self.paid[result.request_id] = bool(result.success)
        for candidate in candidates:
            self.geometric.append((self.requests[candidate.request_id]["segments"], candidate.spherical_cells))
