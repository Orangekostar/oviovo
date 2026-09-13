#!/usr/bin/env python3
"""Recover missing complete-query records by reusing verified native segment inputs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.run_static_replica_scene import verify_native_manifest
from scripts.evaluation.run_ovimap_native import (
    default_config, _new_attempt, build_scene_commands, preflight, _run_command,
    _mapping_env, audit_scene, write_scene_manifest)
from src.static_ovmap.cache_io import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-manifest', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    parent = verify_native_manifest(args.parent_manifest)
    scene = parent['scene']
    config = replace(default_config(args.run_root), capture_query_history=True)
    attempt = _new_attempt(args.run_root, scene)
    parent_root = args.parent_manifest.parent
    reused = {}
    for name in ('frontend', 'geometric_segments'):
        source = parent_root/name
        files = sorted(f for f in source.iterdir() if f.is_file())
        if not files:
            raise ValueError('missing verified segment inputs: '+str(source))
        reused[name] = {str(f): sha256_file(f) for f in files}
        (attempt/name).rmdir()  # only the just-created empty destination
        (attempt/name).symlink_to(source.resolve(), target_is_directory=True)
    commands = build_scene_commands(config, scene=scene, start=0, end=2000, step=10, attempt_root=attempt)
    current = preflight(config, commands, scene=scene)
    if current['sources'] != parent['preflight']['sources']:
        raise ValueError('native build sources differ from the segment-producing parent')
    receipt = {'status': 'RUNNING', 'reason': 'parent did not export complete executed-query history',
               'parent_manifest': str(args.parent_manifest),
               'parent_manifest_sha256': sha256_file(args.parent_manifest),
               'reused_segment_sha256': reused, 'command': sys.argv,
               'selection_rule': 'complete-query capture, not best evaluation score',
               'source_sha256': sha256_file(__file__),
               'cache_scope': 'warm CropFormer and geometric masks; fresh mapping and image VLM queries'}
    record_path = attempt/'reuse_receipt.json'
    record_path.write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        mapping = _run_command(commands.mapping, cwd=config.ovimap_root, env=_mapping_env(config),
                               log=attempt/'logs/mapping.log')
        for files in reused.values():
            if any(sha256_file(f) != digest for f, digest in files.items()):
                raise ValueError('reused segment inputs changed during mapping')
        audit = audit_scene(commands)
        if audit['status'] != 'PASS':
            raise ValueError('native audit failed')
        frontend = {**parent['frontend'], 'reused': True, 'incremental_wall_seconds': 0}
        geometry = {**parent['mapping']['geometry'], 'reused': True, 'incremental_wall_seconds': 0}
        manifest_path = write_scene_manifest(commands, scene=scene,
            stage='room' if scene == 'room0' else 'mapping',
            state='ROOM0_PASS' if scene == 'room0' else 'MAPPING_PASS',
            preflight_record=current, frontend_record=frontend,
            mapping_record={'status': 'PASS', 'geometry': geometry, 'mapping': mapping}, audit_record=audit)
        receipt['status'] = 'COMPLETE'
        record_path.write_text(json.dumps(receipt, indent=2)+'\n')
        manifest = json.loads(manifest_path.read_text())
        manifest['segment_reuse'] = {'receipt': str(record_path), 'sha256': sha256_file(record_path),
            'cost_rule': 'parent frontend/geometry timestamps are provenance, not new elapsed work; incremental cost is zero'}
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print(manifest_path)
    except Exception as error:
        receipt.update(status='FAILED', error=str(error))
        record_path.write_text(json.dumps(receipt, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
