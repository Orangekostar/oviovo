"""Explicit PDF source selection and immutable supplied-transcription comparison."""

import argparse
import copy
from pathlib import Path
import re
import subprocess

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .protocol import PACKAGE


SOURCE_URLS = {
    "cvf_final": "https://openaccess.thecvf.com/content/CVPR2026/papers/"
                 "Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.pdf",
    "arxiv_v1": "https://arxiv.org/pdf/2603.26541v1",
}
METHOD_NAMES = {"Mask3D+OM3D": "Mask3D + OpenMask3D",
                "Segment3D+OM3D": "Segment3D + OpenMask3D", "OVO-SLAM": "OVO-SLAM"}
SOURCE_COLUMNS = ("miou", "macc", "ap25", "ap50", "apall")


def parse_table3(text):
    headers = list(re.finditer(r"Method\s+Online\s+mIoU\s+mAcc\s+AP25\s+AP50\s+APall", text))
    if len(headers) != 1:
        raise ValueError("PDF extraction must identify exactly one Table3 column header")
    tail = text[headers[0].end():]
    caption = re.search(r"Table\s+3\.", tail)
    if caption is None:
        raise ValueError("PDF extraction omitted the Table3 caption boundary")
    region = tail[:caption.start()]
    pattern = (r"^\s*(Mask3D\+OM3D|Segment3D\+OM3D|OVO-SLAM)\s+\S+\s+"
               r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)(?=\s|$)")
    matches = list(re.finditer(pattern, region, re.MULTILINE))
    expected = list(METHOD_NAMES) * 2
    if len(matches) != 6 or [row[1] for row in matches] != expected:
        raise ValueError("PDF Table3 requires exactly six ordinary source rows in their authored order")
    rows = [{"method": name} for name in METHOD_NAMES.values()]
    for position, match in enumerate(matches):
        values = list(map(float, match.groups()[1:]))
        if any(not 0 <= value <= 100 for value in values):
            raise ValueError("PDF Table3 metric leaves its published percentage range")
        rows[position % 3][("replica8", "scannet_cf18")[position // 3]] = dict(
            zip(SOURCE_COLUMNS, values, strict=True))
    return rows


def reconcile_tables(provided, final_rows, arxiv_rows):
    if final_rows != arxiv_rows:
        raise ValueError("the final CVF PDF and arXiv v1 PDF disagree; explicit source selection must be reconsidered")
    if [row["method"] for row in provided["rows"]] != [row["method"] for row in final_rows]:
        raise ValueError("source audit changed the three attributed author methods")
    differences = []
    for supplied, verified in zip(provided["rows"], final_rows, strict=True):
        for cohort in ("replica8", "scannet_cf18"):
            for metric in SOURCE_COLUMNS:
                before, after = supplied[cohort][metric], verified[cohort][metric]
                if before != after:
                    differences.append({"method": verified["method"], "cohort": cohort, "metric": metric,
                        "provided_percent": before, "chosen_pdf_percent": after})
    return {"status": "AUTHOR_REPORTED_NOT_REPRODUCED", "source_url": SOURCE_URLS["cvf_final"],
        "source_version": "CVPR 2026 CVF final PDF; independently checked against arXiv v1 PDF",
        "source_table": 3, "source_pdf_page": 6, "source_printed_page": 12611,
        "units": "percent", "published_decimal_places": 1, "rows": copy.deepcopy(final_rows),
        "provided_transcription": copy.deepcopy(provided), "differences_from_provided_transcription": differences,
        "provided_file_overwritten": False, "all_30_pdf_values_agree": True,
        "selection_reason": "Choose the visually verified author PDF representation explicitly. Both PDFs agree; "
                            "the supplied HTML transcription differs. No version averaging or column swapping.",
        "provided_anomaly_note": "The supplied ScanNet Mask3D AP25<AP50 note is retained in the original. "
                                 "The verified PDF values are AP25=10.4 and AP50=8.0; no correction was inferred.",
        "full_scorer_protocol_independently_verified": False,
        "external_protocol_status": "AUTHOR_PROTOCOL_AS_REPORTED",
        "author_scannet_scene_cohort_verified_equal_to_CF18": False,
        "cohort_keys_are_table_placement_only": True,
        "input_supervision_notes": {"offline_rows": "Mask3D/Segment3D mesh masks with OpenMask3D RGB-D labeling; "
            "pretrained components, not independently reproduced here.",
            "OVO-SLAM": "Ordinary online setting, excluding the 30-fps variant."},
        "publishing_rule": provided["publishing_rule"]}


def verify_reference(output_root):
    root = Path(output_root) / "external"
    index = ConsumptionIndex(root / "input_verifications.json")
    supplied_path = PACKAGE / "literature_reference.json"
    supplied_identity = index.identity(supplied_path)
    sources, parsed = {}, {}
    for name, url in SOURCE_URLS.items():
        pdf = root / "sources" / ("ovi_map_" + name + ".pdf")
        pdf_identity = index.identity(pdf)
        text_path = pdf.with_suffix(".txt")
        subprocess.run(["pdftotext", "-layout", str(pdf), str(text_path)], check=True)
        parsed[name] = parse_table3(text_path.read_text())
        sources[name] = {"url": url, "pdf": pdf_identity, "extracted_text": index.identity(text_path)}
    result = reconcile_tables(read(supplied_path), parsed["cvf_final"], parsed["arxiv_v1"])
    image = root / "sources/ovi_map_cvf_table3_page.png"
    if not image.is_file():
        raise ValueError("external publication requires the inspected PDF table-page image")
    result.update(sources=sources, supplied_file=supplied_identity, visual_table_page=index.identity(image),
                  inputs=index.entries(), producer=index.identity(__file__))
    result["identity"] = canonical_digest(result)
    atomic_write_json(root / "reference.json", result)
    index.write_memo(root / "input_verifications.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = verify_reference(args.output_root)
    print("External PDFs agree; provided transcription differences:",
          len(result["differences_from_provided_transcription"]), result["identity"], flush=True)
