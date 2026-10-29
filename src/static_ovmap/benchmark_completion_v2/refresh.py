"""Explicit asset refresh preserves completed numerical producers and outputs."""
from pathlib import Path

from .common import read, verified, write, canonical_digest, digest_file

ORDER = ('protocol','method_lock','legacy_predict','legacy_score','benchmark_inputs','baselines',
         'benchmark_predict','benchmark_score','statistics','tables','review','publish')


def affected_phases(old, new, *, mapping_changed=False):
    phases = set()
    old_alias = old.get('datasets', {}).get('scannet200_val', {}).get('label_mapping_path')
    new_alias = new.get('datasets', {}).get('scannet200_val', {}).get('label_mapping_path')
    if old_alias != new_alias:
        phases.update(('protocol','legacy_predict','legacy_score'))
    if old.get('external', {}) != new.get('external', {}):
        phases.update(('baselines','tables','review','publish'))
    if old.get('datasets', {}) != new.get('datasets', {}):
        phases.update(('benchmark_inputs','baselines','benchmark_predict','benchmark_score',
                       'statistics','tables','review','publish'))
    if (mapping_changed or old.get('source_assets', {}) != new.get('source_assets', {})
            or old.get('actual_user_or_scheduler_quota', {}) != new.get('actual_user_or_scheduler_quota', {})):
        phases.update(ORDER)
    if old.get('publication', {}) != new.get('publication', {}):
        phases.add('publish')
    return [phase for phase in ORDER if phase in phases]


def bind_assets(args):
    from .binding import bind
    root = Path(args.output_root).resolve()
    old = verified(root / 'binding.json')
    pending_path = root / 'asset_refresh_pending.json'
    pending = verified(pending_path) if pending_path.is_file() else None
    if pending:
        descriptor = pending['previous_binding']
        if digest_file(descriptor['path']) != descriptor['sha256']:
            raise ValueError('Pending asset refresh lost its original binding bytes')
        old = verified(descriptor['path'])
    if not args.asset_config:
        raise ValueError('--refresh-assets requires an actual asset-config file')
    assets = read(args.asset_config)
    if pending and canonical_digest(assets) != pending['new_asset_identity']:
        raise ValueError('Finish the pending explicit refresh before supplying different assets')
    if assets.get('gpu_lock') and assets['gpu_lock'] != old['asset_config'].get('gpu_lock',
            f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{old['gpu']}.lock"):
        raise ValueError('Asset refresh cannot replace the inherited single GPU lock')
    mapping = read(args.path_map) if args.path_map else assets.get('path_map', old['path_map'])
    if pending and mapping != pending['path_map']:
        raise ValueError('Pending refresh path mapping changed')
    protocol = verified(root / 'protocol_lock.json') if (root / 'protocol_lock.json').is_file() else None
    if pending and pending['previous_protocol']:
        descriptor = pending['previous_protocol']
        if digest_file(descriptor['path']) != descriptor['sha256']:
            raise ValueError('Pending asset refresh lost its original protocol bytes')
        protocol = verified(descriptor['path'])
    candidate = None
    if protocol:
        # A new raw/GT/checkpoint location is an asset update. Replacing the
        # frozen scene/vocabulary/evaluator definition is a scientific repair.
        fields = {'scannet200_val': {'official_split_path':'split_scan','official_evaluator_path':'evaluator_scan'},
                  'replica48': {'official_eight_scene_list':'split_replica','class48_mapping_path':'classes_replica',
                                'official_evaluator_path':'evaluator_replica'}}
        for benchmark, keys in fields.items():
            for field, source in keys.items():
                supplied = assets.get('datasets',{}).get(benchmark,{}).get(field)
                if supplied and digest_file(supplied) != protocol['official_sources'][source]['sha256']:
                    raise ValueError('Asset refresh cannot silently replace the frozen scientific protocol: '+field)
    phases = affected_phases(old['asset_config'], assets, mapping_changed=mapping != old['path_map'])
    identity = canonical_digest(dict(old=old['asset_identity'],new=assets,path_map=mapping))
    history = root / 'history/asset_refresh' / identity
    if protocol:
        from .semantic_refresh import check_alias, prepare_protocol
        preview = dict(old, asset_config=assets, path_map=mapping)
        try:
            check_alias(preview, protocol)
        except ValueError:
            # A legacy controller may still hold the old protocol in memory.
            # Explicit refresh must run at an idle controller boundary.
            import os
            for process in Path('/proc').iterdir():
                if not process.name.isdigit() or int(process.name) == os.getpid():
                    continue
                try:
                    command = (process / 'cmdline').read_bytes().split(b'\0')
                except (OSError, ProcessLookupError):
                    continue
                if b'--output-root' in command and any(b'run_ovmap_benchmark_completion_v2.py' in arg for arg in command):
                    position = command.index(b'--output-root') + 1
                    if position < len(command) and Path(os.fsdecode(command[position])).resolve() == root:
                        raise ValueError('Semantic asset refresh requires the existing controller to finish its active boundary')
            candidate = prepare_protocol(preview, protocol, history / 'protocol')
            phases = [p for p in ORDER if p in set(phases) | {'protocol','legacy_predict','legacy_score',
                'benchmark_predict','benchmark_score','statistics','tables','review','publish'}]
    write(history / 'binding.json', old)
    previous_protocol = None
    if protocol:
        previous_path = history / 'protocol_before.json'
        write(previous_path, protocol)
        previous_protocol = dict(path=str(previous_path), sha256=digest_file(previous_path))
    write(pending_path, dict(status='RUNNING', previous_binding=dict(path=str(history / 'binding.json'),
        sha256=digest_file(history / 'binding.json')), previous_protocol=previous_protocol,
        new_asset_identity=canonical_digest(assets), path_map=mapping, affected_phases=phases))
    result = bind(args)
    if result['public_A_identity'] != old['public_A_identity'] or result['gpu'] != old['gpu']:
        # A failed identity check remains reviewable; restore the original live
        # binding, without changing any calibration or model products.
        write(root / 'binding.json', old)
        raise ValueError('Asset refresh changed the original calibration population or GPU')
    if candidate is not None:
        write(root / 'protocol_lock.json', candidate)
    inventory = {}
    raw_root = assets.get('datasets',{}).get('scannet200_val',{}).get('raw_root')
    if raw_root and protocol:
        for scene in protocol['scopes']['scannet200_val']['scenes']:
            files = {}
            for suffix in ('.sens','_vh_clean_2.ply','.txt','.aggregation.json',
                           '_vh_clean_2.0.010000.segs.json','_vh_clean_2.labels.ply'):
                candidates = [Path(raw_root)/scene/(scene+suffix),Path(raw_root)/'scans'/scene/(scene+suffix)]
                found = next((p for p in candidates if p.is_file()),None)
                if found:
                    files[suffix] = dict(path=str(found),bytes=found.stat().st_size)
            inventory[scene] = dict(files=files)
        write(root/'assets/refreshed_scannet_inventory.json',dict(status='DONE',scenes=inventory,
            authorized_source_root=str(raw_root),asset_refresh=identity,GT_labels_opened=False,
            search='exact official scene filenames in two declared root layouts; metadata only'))
    terminal = root/'terminal.json'
    if terminal.is_file():
        write(history/'terminal.json',verified(terminal))
        terminal.unlink()
    write(root/'asset_refresh.json',dict(status='DONE',refresh_identity=identity,old_binding=old['identity'],
        new_binding=result['identity'],affected_phases=phases,completed_CPU_calibration_preserved=True,
        completed_job_outputs_deleted=0,
        protocol_lock_preserved=None if protocol is None or candidate is not None else protocol['identity'],
        semantic_protocol_refresh=None if candidate is None else dict(previous=protocol['identity'],
            current=candidate['identity'], history=str(history / 'protocol')),
        completed_scientific_outputs_are_validated_on_reuse=True,history=str(history)))
    pending_path.unlink()
    return result
