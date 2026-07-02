import numpy as np

from src.core.data_structures import ProposalSoftScores, RefinedProposal2D
from src.modules.proposal_cleanup import ProposalCleanupModule


def _proposal(proposal_id, mask, confidence, objectness, label=""):
    return RefinedProposal2D(
        proposal_id=proposal_id,
        mask=mask,
        bbox_xyxy=np.array([0, 0, mask.shape[1], mask.shape[0]], dtype=np.float32),
        area=int(mask.sum()),
        confidence=confidence,
        soft_scores=ProposalSoftScores(objectness_score=objectness),
        metadata={"anchor_class_name": label, "anchor_confidence": confidence},
    )


def test_exclusivity_subtracts_lower_priority_overlap():
    high = np.zeros((6, 6), dtype=bool)
    high[1:5, 1:5] = True
    low = np.zeros((6, 6), dtype=bool)
    low[2:6, 2:6] = True
    proposals = [
        _proposal(1, low, confidence=0.4, objectness=0.5, label="sofa"),
        _proposal(2, high, confidence=0.9, objectness=0.9, label="cup"),
    ]

    cleaned = ProposalCleanupModule({"enabled": True, "min_area_after_cleanup": 1}).process(proposals)
    by_id = {proposal.proposal_id: proposal for proposal in cleaned}

    assert np.count_nonzero(by_id[1].mask & by_id[2].mask) == 0
    assert by_id[2].area == 16
    assert by_id[1].area < 16


def test_cleanup_drops_masks_below_min_area():
    mask_a = np.zeros((4, 4), dtype=bool)
    mask_a[:, :] = True
    mask_b = mask_a.copy()
    proposals = [
        _proposal(1, mask_a, confidence=0.9, objectness=0.9),
        _proposal(2, mask_b, confidence=0.1, objectness=0.1),
    ]

    cleaned = ProposalCleanupModule({"enabled": True, "min_area_after_cleanup": 4}).process(proposals)

    assert [proposal.proposal_id for proposal in cleaned] == [1]
