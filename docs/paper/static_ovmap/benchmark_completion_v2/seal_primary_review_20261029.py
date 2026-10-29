"""Seal the primary's documented decisions against current, closed evidence.

No model, scorer, dataset conversion, or publication is executed here.
"""
import argparse
from collections import Counter
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]
from static_ovmap.benchmark_completion_v2.common import digest_file, verified, write
from static_ovmap.benchmark_completion_v2.lifecycle import producer_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recovery-root', type=Path, required=True)
    args = parser.parse_args()
    root, recovery = args.output_root.resolve(), args.recovery_root.resolve()
    target = root / 'review/primary_final.json'
    assert not target.exists(), 'This final-primary seal is one-shot'
    evidence = {}

    def check(base, relative, expected):
        path = base / relative
        value = verified(path)
        assert value['identity'] == expected, relative
        evidence[str(path)] = dict(identity=value['identity'], sha256=digest_file(path))
        return value

    state = check(recovery, 'source_transition_20261029/actual_execution_state.json',
                  '95960ff163568ff4814ae591a2ca04761bf99fcd3004aa7e3b18908fd2437f18')
    assert len(state['current_execution_files']) == 99
    for name, row in state['current_execution_files'].items():
        path = Path(name)
        stat = path.stat()
        assert (digest_file(path), stat.st_size, stat.st_mtime_ns) == (
            row['sha256'], row['bytes'], row['mtime_ns']), name
    check(recovery, 'source_transition_20261029/refreshed_standard_review.json',
          'b8772a1cf542fda8b245c2fd96b82eadefa25277deabefb63e551dc8e380284b')
    check(recovery, 'distributed_diagnostics_20261029/completed_execution_review.json',
          'fbf9c3b5d18dd9daee854b5f25d086ca7bb09b2e3c7619697ae0547a80c1014c')
    check(root, 'review/G1_old_RAW_exact.json',
          '120d9251ff20f18d622bd7a666476d38020058239ae94d8a7e7cff31822def79')
    check(root, 'review/G1_real_numeric_audit.json',
          '6ed752da46f53834931983fa633b8adf2f611a9c368b49d486e274ac16c90a33')
    check(root, 'model_lock.json',
          '69592d6c35139cdd27f25ff16cfed1b0d1d5e3c0e659c59c77bca77ecfda6eae')
    check(root, 'review/G3_global_actual.json',
          '4695f742709c1e437bb1df92ca21607143495ae6cc14c8e421007b424f833ff7')
    check(root, 'review/G3_evaluator_auxiliary_actual_parity.json',
          'af591e333fd61576ee32897870e13f643216f8819247f361094b81d67e1875f7')
    g4 = check(root, 'review/G4_actual_aggregation.json',
               '4387818d3a08919c2e96e4c97a11698351065f9d726042cd8d7aacaf47ac3b67')
    check(root, 'review/visual_QA.json',
          '8bc17ad1f9489bf3b8a0c33f6bc33f5ffdecb989b69fa38517fbc4004b84218b')
    visual = check(root, 'review/visual_supplement_final.json',
                   'a037fc67df725a3f86c2ff7603c079839c6ac6a07b4ab0e1c620915e0cfa002e')
    check(root, 'supplement/preview_receipt.json',
          '15f0e378e612e98ace31fdb74f84ff03db3b942bd8da0e3d7e91a5269f7bed6b')
    check(root, 'review/final_supplement/machine_review.json',
          '8f7c43caf4ca2d51639a50ea26e48955bc28a545f14be3ca4d1afc7cd2860cfe')
    profile = check(root, 'publication/execution_provenance.json',
                    '9efbb8a88f88c72026a7d7618c61de0036ecb11d1c0862c9bb457a1ab86b2063')
    assert len(profile['files']) == 4543 and not profile['profile_requires_refresh_before_final_publication']
    tables = verified(root / 'tables.json')
    store = verified(root / 'result_store.json')
    assert tables['identity'] == g4['tables']
    assert store['identity'] == g4['result_store']
    assert g4['numeric_cells_checked'] == 36185 and g4['actual_T1_T2_U11_keys_equal']
    assert visual['source_rows'] == 6019 and visual['total_pages'] == 163
    assert visual['all_163_page_contacts_viewed'] and visual['all_eight_first_pages_viewed']
    statistics = verified(root / 'statistics/scannet200_val.json')
    assert statistics['identity'] == '7132c270455b2ff1f0d7932ac034b0fb85bf55443000ee367f0c5791bf2553f2'
    assert len(statistics['scenes']) == 312 and len(statistics['families']) == 142
    assert set(statistics['comparisons']) == {'U11-U10', 'U11-U01'}
    assert all(row['valid_draws'] == 1000 and row['undefined_draws'] == 0
               for row in statistics['comparisons'].values())
    diagnostics = verified(root / 'diagnostics/scannet200_val/summary.json')
    assert diagnostics['identity'] == '5339c06454babbfdf063cbe8ba5dabe40483d35bbbdf21e6a1440223a523b4fe'
    assert diagnostics['full_scene_coverage'] and diagnostics['completed_scenes'] == 312
    coverage = verified(root / 'review/coverage.json')
    assert coverage['identity'] == '794746573be97fd3178a744a1cc0230b8b4625700c1b1155c84456af853f4f8f'
    assert coverage['standard']['scannet200_val']['completed_cells'] == 2496
    assert coverage['standard']['replica48']['completed_cells'] == 0
    remaining = verified(root / 'remaining_jobs.json')
    assert len(remaining['jobs']) == 129 and not any(row['runnable'] for row in remaining['jobs'])
    assert Counter(row['kind'] for row in remaining['jobs']) == {
        'STANDARD_SCENE_CELL': 64, 'LEGACY_SCENE_CELL': 32,
        'EXTERNAL_SCENE_METHOD': 32, 'PAIRED_BOOTSTRAP': 1}
    roles = {}
    for name, expected in {
        'refreshed_benchmark_inputs_20261029': 20, 'refreshed_benchmark_predict_20261029': 20,
        'refreshed_benchmark_score_20261029': 20, 'refreshed_statistics_20261029': 20,
        'refreshed_baselines_20261029': 20, 'refreshed_tables_20261029_v2': 0,
        'refreshed_review_20261029': 0, 'distributed_diagnostics_20261029_v2': 0,
        'terminal_provenance_20261029_v2': 0,
    }.items():
        folder = recovery / 'roles' / name
        actual, sync = verified(folder / 'actual_exit.json'), verified(folder / 'synchronized.json')
        assert actual['actual_child_exit'] == expected
        assert sync['status'] == 'DONE' and sync['source_unchanged']
        assert sync['actual_exit'] == actual['identity']
        roles[name] = dict(actual=actual['identity'], synchronized=sync['identity'], actual_exit=expected)
    git = lambda *argv: subprocess.check_output(['git', *argv], cwd=REPO, text=True).strip()
    assert git('branch', '--show-current') == 'research/ovmap-benchmark-completion-v2'
    assert git('rev-parse', 'HEAD') == '9bbe040570ecbaec854e6d8aa94ba82a3a1afa2a'
    assert not git('diff', '--cached', '--name-only')
    docs = Path(__file__).parent
    text = (docs / 'FINAL_REQUIREMENT_REVIEW.md').read_text()
    assert re.findall(r'^\| R(\d{2}) /', text, re.M) == [f'{n:02d}' for n in range(1, 40)]
    document_names = ['RESULTS.md', 'HANDOFF.md', 'REVIEW.md', 'CLAIMS.md',
                      'FINAL_REQUIREMENT_REVIEW.md', Path(__file__).name]
    documents = {name: digest_file(docs / name) for name in document_names}
    before = root / 'review/G4_coverage_before_primary_20261029.json'
    assert not before.exists()
    shutil.copy2(root / 'review/coverage.json', before)
    primary = write(target, dict(status='DONE', requirements_review_complete=True,
        reviewed_producer_identity=producer_identity(), result_store_identity=store['identity'],
        tables=tables['identity'], coverage_before_primary=coverage['identity'], remaining_jobs=remaining['identity'],
        requirements=list(range(1, 40)), original_00_chapters=list(range(1, 18)),
        companion_03_sections=list('ABCDEFG'), companion_04_sections=list(range(1, 7)),
        primary_decisions_document='FINAL_REQUIREMENT_REVIEW.md', documents=documents,
        evidence=evidence, actual_closed_roles=roles, execution_files_verified=99,
        software_complete=False, software_incomplete_reason='REPLICA_PORT_FIRST_ACTUAL_G3_PENDING',
        claim_supported=False, full_objective_complete=False, no_runnable_scientific_jobs=True,
        publication_authorized_for_partial_results=True, publication_not_yet_verified=True,
        source_review_method='PRIOR_PRIMARY_COMPONENT_REVIEWS_PLUS_EXACT_V3_TRANSITION_AND_CURRENT_SOURCE_GUARD',
        real_computation_distinguished_from_synthetic_and_author_references=True,
        replica_3d_is_a_directory_name_not_a_separate_dataset=True,
        equivalent_raw_Replica_conversion_requires_verified_instance_GT_and_evaluation_domain=True,
        old_N0_unchanged=True, new_training_updates=0, synthetic=False))
    coverage.update(final_primary_review_complete=True, primary_final_review=primary['identity'])
    updated = write(root / 'review/coverage.json', coverage)
    assert updated['completion_flags'] == verified(before)['completion_flags']
    assert verified(root / 'result_store.json')['identity'] == store['identity']
    print('PRIMARY_REVIEW_SEALED', primary['identity'], 'COVERAGE', updated['identity'], flush=True)


if __name__ == '__main__':
    main()
