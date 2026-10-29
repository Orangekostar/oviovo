"""Completed real job consumption; planning ranges never become measured FPS."""
from pathlib import Path
import statistics
import subprocess

from .common import REPO, Resolver, canonical_digest, digest_file, verified, write, read


def _phase_binding(root, binding, entry, protocol):
    """Follow explicit asset-refresh history to the unchanged scientific phase."""
    if entry['scientific_binding'] == binding['identity']:
        return protocol['identity'], None
    receipt = root / 'asset_refresh.json'
    if not receipt.is_file() or (root / 'asset_refresh_pending.json').exists():
        raise ValueError('Parallel phase requires a completed linked asset refresh')
    refresh = verified(receipt)
    if (refresh['status'] != 'DONE' or refresh['new_binding'] != binding['identity']
            or verified(root / 'binding.json') != binding):
        raise ValueError('Parallel phase asset refresh does not bind the current run')
    history_root = root / 'history/asset_refresh'
    current, chain, seen = binding, [], set()
    ignored = {'identity', 'asset_config', 'asset_identity', 'input_identity', 'path_map'}
    core = lambda value: {k: v for k, v in value.items() if k not in ignored}
    archives = sorted(history_root.glob('*/binding.json'))
    while current['identity'] != entry['scientific_binding']:
        if current['identity'] in seen:
            raise ValueError('Parallel phase asset-refresh history contains a cycle')
        seen.add(current['identity'])
        if current.get('asset_identity') != canonical_digest(current.get('asset_config', {})):
            raise ValueError('Parallel phase current asset identity differs')
        matches = []
        for path in archives:
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError('Parallel phase refresh history cannot follow symlinks')
            prior = verified(path)
            if prior['identity'] == current['identity']:
                continue
            if prior.get('asset_identity') != canonical_digest(prior.get('asset_config', {})):
                continue
            transition = canonical_digest(dict(old=prior['asset_identity'],
                new=current['asset_config'], path_map=current.get('path_map', {})))
            relocated = Resolver(current.get('path_map', {})).rewrite(core(prior))
            if path.parent.name == transition and relocated == core(current):
                matches.append((path, prior))
        if len(matches) != 1:
            raise ValueError('Parallel phase has no unique scientifically unchanged refresh ancestor')
        path, prior = matches[0]
        if not chain and (Path(refresh['history']) != path.parent
                or refresh['refresh_identity'] != path.parent.name
                or refresh['old_binding'] != prior['identity']):
            raise ValueError('Parallel phase latest refresh history is not linked')
        chain.append(dict(history=str(path.parent), previous_binding=prior['identity'],
            next_binding=current['identity'], previous_binding_sha256=digest_file(path),
            declared_path_map=current.get('path_map', {}),
            non_asset_binding_fields_equal=core(prior) == core(current)
                and prior.get('path_map', {}) == current.get('path_map', {})))
        current = prior
    old_protocol = verified(Path(chain[-1]['history']) / 'protocol_before.json')
    if old_protocol['identity'] != entry['protocol']:
        raise ValueError('Parallel phase original protocol is absent from refresh history')
    if old_protocol['identity'] != protocol['identity']:
        scopes = [p.get('scopes', {}).get('scannet200_val') for p in (old_protocol, protocol)]
        evaluators = [p.get('official_sources', {}).get('evaluator_scan') for p in (old_protocol, protocol)]
        if (scopes[0] is None or scopes[0] != scopes[1] or not all(evaluators)
                or any(evaluators[0][k] != evaluators[1][k] for k in ('sha256', 'bytes'))):
            raise ValueError('Parallel phase ScanNet scientific protocol changed')
    return old_protocol['identity'], dict(phase_binding=current['identity'],
        current_binding=binding['identity'],
        non_asset_binding_fields_equal=all(row['non_asset_binding_fields_equal'] for row in chain),
        non_asset_binding_fields_equal_after_declared_path_rewrite=True,
        phase_protocol=old_protocol['identity'], current_protocol=protocol['identity'],
        exact_scannet_scope_preserved=True, asset_refresh_chain=chain)


def _runtime_allocation(root, binding, hardware):
    """Bind configuration limits without rewriting historical device inventory."""
    relative = 'bootstrap/parallel_scenes/entry.json'
    path = root / relative
    if not path.is_file():
        return dict(source='resources/hardware.json', identity=hardware['identity'],
            scope='HARDWARE_SNAPSHOT_CONFIGURATION_LIMITS_NOT_MEASURED_UTILIZATION',
            physical_GPUs=[int(hardware['selected_physical_GPU'])],
            maximum_GPU_workers=hardware['maximum_GPU_workers'],
            maximum_CPU_scorers=hardware['maximum_CPU_scorers'], BLAS_threads=hardware['BLAS_threads'])
    entry = verified(path)
    protocol, carry = _phase_binding(root, binding, entry, verified(root / 'protocol_lock.json'))
    expected = dict(scientific_binding=entry['scientific_binding'] if carry else binding['identity'],
        protocol=protocol,
        model=verified(root / 'model_lock.json')['identity'],
        source_sha256=digest_file(REPO / 'scripts/evaluation/run_ovmap_parallel_scenes.py'),
        scientific_settings_changed=False)
    for key, value in expected.items():
        if entry.get(key) != value:
            raise ValueError('Parallel resource phase differs from this run: ' + key)
    return dict(source=relative, identity=entry['identity'],
        scope='PARALLEL_SCENE_PHASE_CONFIGURATION_LIMITS_NOT_MEASURED_UTILIZATION',
        physical_GPUs=entry['gpus'], maximum_GPU_workers=entry['max_scene_workers'],
        maximum_CPU_scorers=entry['max_own_CPU_scorers'] + entry['existing_OVI_CPU_scorers'],
        BLAS_threads=hardware['BLAS_threads'], **({'binding_carry': carry} if carry else {}))


def build(binding):
    from .runtime import gpu_seconds
    root = Path(binding['output_root'])
    hardware_path = root / 'resources/hardware.json'
    if not hardware_path.is_file():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), 'UNKNOWN')
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True)
        write(hardware_path, dict(status='DONE', CPU=cpu, GPU_inventory=gpu.stdout.splitlines(),
            GPU_query_exit=gpu.returncode, selected_physical_GPU=str(binding['gpu']),
            maximum_GPU_workers=1, maximum_CPU_scorers=3, BLAS_threads=1))
    hardware = verified(hardware_path)
    allocation = _runtime_allocation(root, binding, hardware)
    counters = dict(new_training_updates=0, calibration_GPU_head_calls=0,
        calibration_RGB_encodings=0, standard_CropFormer_frames=0,
        standard_FC_image_encodings=0, standard_FC_region_poolings=0,
        standard_MV_head_calls=0, standard_N_crop_encodings=0, standard_Q_crop_encodings=0)
    references, components, pipeline_times = {}, {}, []
    for label, relative, field in (
            ('engineering_MV_head_calls', 'model_lock.json', 'actual_head_calls'),
            ('legacy_new_MV_head_calls', 'legacy/prediction_lock.json', 'new_head_calls_this_invocation')):
        path = root / relative
        if path.is_file():
            value = verified(path)
            counters[label] = value[field]
            references[relative] = value['identity']
    for frame_path in sorted((root / 'benchmark/frames').glob('*/*/frame_plan.json')):
        frame = read(frame_path)
        benchmark, scene = frame_path.parent.parent.name, frame_path.parent.name
        bundle_path=root/'benchmark/bundles'/benchmark/scene/'bundle.json'
        bundle=verified(bundle_path) if bundle_path.is_file() else None
        key = benchmark + ':' + scene
        rows = {}
        for name in ('cropformer', 'native_map', 'F_MV1000', 'native_query'):
            path = root / 'jobs/benchmark' / benchmark / scene / name / 'receipt.json'
            if not path.is_file():
                continue
            job = verified(path)
            if job['status'] != 'DONE':
                continue
            rows[name] = dict(worker_wall_seconds=job['elapsed_seconds'], receipt=job['identity'],
                confidence='ACTUAL_COMPLETED_CHILD_WALL_TIME', includes_CPU_and_model_load=True,
                module_only_latency_seconds=None, scheduled_threads=1 if name != 'native_map' else 3,
                peak_GPU_allocated_bytes=None)
            references[str(path.relative_to(root))] = job['identity']
        if 'cropformer' in rows:
            counters['standard_CropFormer_frames'] += len(frame['frame_ids'])
            rows['cropformer']['processed_frames'] = len(frame['frame_ids'])
        f_path = root / 'benchmark/F' / benchmark / scene / 'receipt.json'
        if f_path.is_file():
            f = verified(f_path)
            for target, source in (('standard_FC_image_encodings', 'physical_image_encodings'),
                                   ('standard_FC_region_poolings', 'physical_region_poolings'),
                                   ('standard_MV_head_calls', 'actual_head_calls')):
                counters[target] += f[source]
            if 'F_MV1000' in rows:
                rows['F_MV1000'].update(processed_images=f['physical_image_encodings'],
                    actual_head_calls=f['actual_head_calls'], module_only_latency_seconds=None)
        nq_path = root / 'benchmark/NQ' / benchmark / scene / 'native_query_receipt.json'
        if nq_path.is_file():
            nq = read(nq_path)
            counters['standard_N_crop_encodings'] += nq['native_physical_image_encodings']
            counters['standard_Q_crop_encodings'] += nq['physical_image_encodings'] - nq['native_physical_image_encodings']
            if 'native_query' in rows:
                rows['native_query'].update(peak_GPU_allocated_bytes=nq['peak_gpu_allocated_bytes'],
                    physical_crop_encodings=nq['physical_image_encodings'],
                    actual_Q_attempts=nq['query_completion']['actual_attempts'])
        complete = len(rows) == 4 and bundle is not None
        total = sum(row['worker_wall_seconds'] for row in rows.values()) + (bundle['elapsed_seconds'] if bundle else 0.)
        if complete:
            pipeline_times.append(total)
        components[key] = dict(stages=rows, all_four_stages_complete=complete,
            source_frames=frame['source_frame_count'], processed_frames=len(frame['frame_ids']),
            bundle_CPU_seconds=None if bundle is None else bundle['elapsed_seconds'],
            sequential_pipeline_seconds=total if complete else None,
            excludes=['raw_download', 'GT_download', 'official_scoring'],
            online_30FPS_measured=False, source_30FPS_deadline_test_run=False)
    protocol = verified(root / 'protocol_lock.json')
    expected = sum(len(scope['scenes']) for scope in protocol['scopes'].values())
    remaining = expected - len(pipeline_times)
    # This deliberately broad extrapolation describes remaining finite work;
    # it is neither a confidence interval nor a GPU-hour quota.
    estimate = None if not pipeline_times else dict(
        completed_formal_scenes=len(pipeline_times), remaining_scene_jobs=remaining,
        observed_median_scene_seconds=statistics.median(pipeline_times),
        remaining_sequential_hours_range=[remaining * min(pipeline_times) * .5 / 3600,
                                          remaining * max(pipeline_times) * 2 / 3600],
        range_definition='0.5x_min_to_2x_max_completed_scene_wall;planning_only;not_CI',
        excludes=['unavailable_data_wait', 'downloads', 'external_baselines', 'official_scoring'])
    ledger = write(root / 'resource_ledger.json', dict(status='DONE', counters=counters,
        completed_job_GPU_worker_wall_upper_bound_seconds=gpu_seconds(root),
        live_job_consumption_not_yet_sealed=True, hardware=hardware['identity'],
        hardware_snapshot_scope='INITIAL_PHASE_DEVICE_INVENTORY_AND_CONFIGURATION',
        runtime_allocation=allocation,
        source_receipts=references, per_scene=components, new_training_updates=0,
        missing_measurements=['whole_pipeline_peak_GPU_memory', 'isolated_FC_module_latency',
                              'isolated_CropFormer_peak_memory', 'true_online_throughput'],
        independent_stage_wall_times_are_not_parallel_throughput=True))
    constraints = root / 'runtime_constraints.json'
    quota = verified(constraints).get('max_gpu_hours') if constraints.is_file() else None
    write(root / 'cost_estimate.json', dict(status='DONE' if estimate else 'BLOCKED_DEPENDENCY',
        resource_ledger=ledger['identity'], estimate=estimate, imposed_GPU_hour_quota=quota,
        source='ACTUAL_FIRST_FORMAL_SCENE;NO_OFFICIAL_HARDWARE_TIME_SUBSTITUTION'))
    return ledger
