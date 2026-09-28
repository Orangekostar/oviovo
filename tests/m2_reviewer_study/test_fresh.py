import hashlib

import pytest


def test_fresh_selection_excludes_physical_family_and_uses_literal_nul_hash():
    from src.static_ovmap.m2_reviewer_study.fresh import choose_families

    rows = [{"scene": f"scene{i:04d}_00", "family": f"scene{i:04d}", "authorized": True,
             "complete": True, "exposure_known": True} for i in range(10)]
    rows.append({**rows[1], "scene": "scene0001_01"})
    rows[2]["exposure_known"] = False
    rows[3]["complete"] = False
    result = choose_families(rows, {"scene0000"})
    eligible = [r for r in rows[:10] if r["family"] not in {"scene0000", "scene0002", "scene0003"}]
    expected = sorted(eligible, key=lambda r: hashlib.sha256(("M2_REVIEW_FRESH_V1\0" + r["family"]).encode()).hexdigest())[:4]
    assert result["chosen"] == [r["scene"] for r in expected]
    assert len({r.split("_")[0] for r in result["chosen"]}) == 4
    assert choose_families(rows[:3], {"scene0000"})["status"] == "FRESH_BLOCKED_INSUFFICIENT_LOCAL_UNEXPOSED_FAMILIES"


def test_fresh_contract_is_bound_to_all_four_audited_rows(tmp_path):
    from src.static_ovmap.composition_study.io import write_once
    from src.static_ovmap.m2_reviewer_study.fresh_execution import contract
    from src.static_ovmap.module_validation.contracts import canonical_digest

    rows = [{"scene": f"scene{i:04d}_00", "authorized": True, "complete": True,
             "exposure_known": True, "exposed": False} for i in range(4)]
    audit = {"binding": "bound", "chosen": [r["scene"] for r in rows], "inventory": rows}
    audit["identity"] = canonical_digest(audit)
    plan = {"status": "FROZEN", "binding": "bound", "scenes": audit["chosen"], "rows": rows,
            "audit_identity": audit["identity"]}
    plan["identity"] = canonical_digest(plan)
    audit_path = tmp_path / "fresh/audits" / (audit["identity"] + ".json")
    write_once(audit_path, audit)
    write_once(tmp_path / "fresh/plan.json", plan)
    write_once(tmp_path / "old/resolved_config.json", {})
    binding = {"output_root": str(tmp_path), "identity": "bound", "composition_root": str(tmp_path / "old")}
    assert contract(binding, "scene0000_00")[2] == rows[0]
    with pytest.raises(ValueError, match="not authorized"):
        contract(binding, "scene0004_00")
    audit_path.write_text(audit_path.read_text().replace('"exposed": false', '"exposed": true'))
    with pytest.raises(ValueError, match="audit identity changed"):
        contract(binding, "scene0000_00")
