"""Fixed recovery planners on current-frame fragments and the own-map prior."""

import numpy as np

from static_ovmap.backbone_wave1.association import _positive_assignment
from static_ovmap.backbone_wave1.fusion import spatial_key


def plan_objects(fragments, depth_m, prior_owner, *, mode):
    if mode not in {"A1", "A2", "A3"}:
        raise ValueError("association arm is not in the fixed recovery protocol")
    depth, prior = np.asarray(depth_m), np.asarray(prior_owner)
    if depth.ndim != 2 or prior.shape != depth.shape:
        raise ValueError("aligned depth and own-map prior required")
    valid = np.isfinite(depth) & (depth > 0) & (depth < 50)
    supports = {}
    for fragment in fragments:
        if fragment.is_thing and fragment.input_group > 0:
            if fragment.mask.shape != depth.shape:
                raise ValueError("fragment support shape differs from the own probe")
            supports.setdefault(fragment.input_group, np.zeros_like(valid))[:] |= fragment.mask & valid
    groups = sorted(supports, key=lambda group: spatial_key(supports[group]))
    global_support = {int(owner): (prior == owner) & valid for owner in np.unique(prior) if owner > 0}
    owners = sorted(global_support, key=lambda owner: spatial_key(global_support[owner]))
    local = np.asarray([supports[group].sum() for group in groups], np.int64)
    glob = np.asarray([global_support[owner].sum() for owner in owners], np.int64)
    intersections = np.zeros((len(groups), len(owners)), np.int64)
    for i, group in enumerate(groups):
        values, counts = np.unique(prior[supports[group]], return_counts=True)
        observed = dict(zip(values.tolist(), counts.tolist(), strict=True))
        intersections[i] = [observed.get(owner, 0) for owner in owners]
    forward = np.divide(intersections, local[:, None], out=np.zeros_like(intersections, np.float64),
                        where=local[:, None] > 0)
    eligible = (intersections >= 100) & (forward > .2)
    if mode == "A1":
        proposed = _positive_assignment(forward - .2, eligible)
    else:
        proposed = {(i, int(np.argmax(np.where(eligible[i], forward[i], -1.))))
                    for i in range(len(groups)) if eligible[i].any()}
    accepted = set(proposed)
    specificity, union_gate = {}, {}
    if mode == "A3":
        accepted = set()
        for i, j in sorted(proposed):
            denominator = int(intersections[i].sum())
            dominance = float(intersections[i, j] / denominator) if denominator else 0.
            competitor = max((float(forward[i, h]) for h in range(len(owners)) if h != j), default=0.)
            margin = float(forward[i, j] - competitor)
            passed = denominator > 0 and dominance >= .8 and margin >= .1
            specificity[str(groups[i])] = {"owner": owners[j], "all_known_intersections": denominator,
                "dominance": dominance, "best_second_margin": margin, "passed": bool(passed)}
            if passed:
                accepted.add((i, j))
        for j in sorted({j for _, j in accepted}):
            members = sorted(i for i, target in accepted if target == j)
            union = np.logical_or.reduce([supports[groups[i]] for i in members])
            covered = int((union & global_support[owners[j]]).sum())
            coverage = covered / int(glob[j]) if glob[j] else 0.
            passed = coverage > .2
            union_gate[str(owners[j])] = {"groups": [groups[i] for i in members],
                "union_intersection": covered, "global_visible_area": int(glob[j]),
                "reverse_union_coverage": coverage, "passed": passed}
            if not passed:
                accepted.difference_update((i, j) for i in members)
    targets = {groups[i]: owners[j] for i, j in accepted}
    actions = [{"local_group": group, "action": "ASSIGN_EXISTING" if group in targets else "USE_NATIVE",
                "existing_owner": targets.get(group)} for group in groups]
    pairs = lambda values: [[groups[i], owners[j]] for i, j in sorted(values)]
    return {"mode": mode, "local_groups": groups, "global_owners": owners,
        "local_areas": local.tolist(), "global_areas": glob.tolist(),
        "intersections": intersections.tolist(), "forward_coverages": forward.tolist(),
        "proposed_pairs": pairs(proposed), "accepted_pairs": pairs(accepted), "actions": actions,
        "specificity": specificity, "union_gate": union_gate,
        "duplicate_owner_followers": len(accepted) - len({j for _, j in accepted}),
        "action_counts": {"USE_NATIVE": len(groups) - len(accepted), "ASSIGN_EXISTING": len(accepted), "CREATE_NEW": 0},
        "prior_global_tie_order": "SPATIAL_VISIBLE_SUPPORT", "GT_input": False}
