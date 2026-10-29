import shutil
import pytest


def test_scene_disk_reserves_twice_next_estimate_above_twenty_gib(tmp_path,monkeypatch):
    from static_ovmap.benchmark_completion_v2 import capacity
    from static_ovmap.benchmark_completion_v2.common import StageFailure
    gib=2**30
    binding=dict(output_root=str(tmp_path),specification=dict(runtime=dict(minimal_disk_reserve_GiB=20)))
    monkeypatch.setattr(capacity,'estimate_scene',lambda *args: dict(estimated_bytes=4*gib,GT_labels_read=False))
    monkeypatch.setattr(shutil,'disk_usage',lambda _: shutil._ntuple_diskusage(100*gib,0,28*gib-1))
    with pytest.raises(StageFailure) as error:
        capacity.check_scene(binding,'scannet200_val','one')
    assert error.value.status=='BLOCKED_RESOURCE'
    monkeypatch.setattr(shutil,'disk_usage',lambda _: shutil._ntuple_diskusage(100*gib,0,28*gib))
    result=capacity.check_scene(binding,'scannet200_val','one')
    assert result['status']=='DONE' and result['required_free_bytes']==28*gib


def test_consumed_gpu_quota_blocks_new_visual_work_but_allows_cpu_completion(tmp_path):
    from static_ovmap.benchmark_completion_v2.work_state import resource_block,scene_state
    from static_ovmap.benchmark_completion_v2.common import write
    binding=dict(output_root=str(tmp_path),specification=dict(runtime=dict(minimal_disk_reserve_GiB=0)))
    write(tmp_path/'runtime_constraints.json',dict(max_gpu_hours=1))
    write(tmp_path/'jobs/benchmark/tiny/one/native_query/receipt.json',dict(status='DONE',
        gpu_worker_launches=1,attempts=[dict(elapsed_seconds=3600)]))
    blocked=resource_block(binding)
    assert blocked['reason']=='ACTUAL_GPU_QUOTA_EXHAUSTED'
    assert not scene_state(binding,'tiny','one',blocked)['runnable']
    for name in ['bundles/tiny/one/bundle.json','F/tiny/one/receipt.json','NQ/tiny/one/sources.json']:
        write(tmp_path/'benchmark'/name,dict(status='DONE'))
    assert scene_state(binding,'tiny','one',blocked)['runnable']


def test_exhausted_download_is_blocked_until_refreshed_original_asset_exists(tmp_path):
    from static_ovmap.benchmark_completion_v2.work_state import scene_state
    from static_ovmap.benchmark_completion_v2.common import write
    binding=dict(output_root=str(tmp_path))
    path=tmp_path/'assets/datasets/scannet/scans/scene1/scene1.sens.download.json'
    write(path,dict(status='BLOCKED_DEPENDENCY',attempts=[{},{}]))
    assert not scene_state(binding,'scannet200_val','scene1')['runnable']
    source=tmp_path/'author.sens'
    source.write_bytes(b'actual asset identity is verified by the acquisition consumer')
    write(tmp_path/'assets/refreshed_scannet_inventory.json',dict(scenes={'scene1':dict(files={'.sens':dict(path=str(source))})}))
    assert scene_state(binding,'scannet200_val','scene1')['runnable']
