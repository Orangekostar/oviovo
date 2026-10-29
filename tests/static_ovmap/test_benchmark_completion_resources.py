"""Real temporary-ledger regression tests; no GPU or science payloads."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

from static_ovmap.benchmark_completion_v2.common import REPO, Resolver, canonical_digest, digest_file, verified, write

candidate=os.environ.get('OVMAP_RESOURCE_CANDIDATE')
if candidate:
    spec=importlib.util.spec_from_file_location(
        'static_ovmap.benchmark_completion_v2._resources_runtime_candidate',candidate)
    resources=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resources)
else:
    from static_ovmap.benchmark_completion_v2 import resources


class ResourceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.binding=write(self.root/'binding.json',dict(output_root=str(self.root),gpu='0'))
        self.hardware_path=self.root/'resources/hardware.json'
        self.hardware=write(self.hardware_path,dict(status='DONE',CPU='fixture CPU',
            GPU_inventory=['0, fixture GPU','2, fixture GPU'],GPU_query_exit=0,
            selected_physical_GPU='0',maximum_GPU_workers=1,maximum_CPU_scorers=3,BLAS_threads=1))
        self.model=write(self.root/'model_lock.json',dict(actual_head_calls=2))
        self.protocol=write(self.root/'protocol_lock.json',dict(scopes={'fixture':dict(scenes=['one'])}))
        self.entry_path=self.root/'bootstrap/parallel_scenes/entry.json'

    def entry(self,**changes):
        value=dict(status='RUNNING',gpus=[0,2],max_scene_workers=2,
            max_own_CPU_scorers=2,existing_OVI_CPU_scorers=1,
            protocol=self.protocol['identity'],model=self.model['identity'],
            scientific_binding=self.binding['identity'],scientific_settings_changed=False,
            source_sha256=digest_file(REPO/'scripts/evaluation/run_ovmap_parallel_scenes.py'))
        value.update(changes)
        return write(self.entry_path,value)

    def fingerprint(self,path):
        return digest_file(path),path.stat().st_mtime_ns

    def test_parallel_phase_is_bound_without_rewriting_initial_hardware(self):
        entry=self.entry();hardware_before=self.fingerprint(self.hardware_path)
        phase_before=self.fingerprint(self.entry_path)
        ledger=resources.build(self.binding)
        self.assertIn('runtime_allocation',ledger)
        allocation=ledger['runtime_allocation']
        self.assertEqual(allocation['maximum_GPU_workers'],2)
        self.assertEqual(allocation['physical_GPUs'],[0,2])
        self.assertEqual(allocation['maximum_CPU_scorers'],3)
        self.assertEqual(allocation['BLAS_threads'],1)
        self.assertEqual(allocation['identity'],entry['identity'])
        self.assertEqual(allocation['source'],'bootstrap/parallel_scenes/entry.json')
        self.assertEqual(ledger['hardware'],self.hardware['identity'])
        self.assertEqual(verified(self.hardware_path)['maximum_GPU_workers'],1)
        self.assertEqual(self.fingerprint(self.hardware_path),hardware_before)
        self.assertEqual(self.fingerprint(self.entry_path),phase_before)
        self.assertEqual(ledger['counters']['engineering_MV_head_calls'],2)
        self.assertEqual(ledger['completed_job_GPU_worker_wall_upper_bound_seconds'],0)
        ledger_before=self.fingerprint(self.root/'resource_ledger.json')
        estimate_before=self.fingerprint(self.root/'cost_estimate.json')
        resumed=resources.build(self.binding)
        self.assertEqual(resumed['identity'],ledger['identity'])
        self.assertEqual(self.fingerprint(self.root/'resource_ledger.json'),ledger_before)
        self.assertEqual(self.fingerprint(self.root/'cost_estimate.json'),estimate_before)

    def test_single_gpu_phase_uses_its_existing_hardware_snapshot(self):
        before=self.fingerprint(self.hardware_path)
        ledger=resources.build(self.binding)
        self.assertIn('runtime_allocation',ledger)
        allocation=ledger['runtime_allocation']
        self.assertEqual(allocation['maximum_GPU_workers'],1)
        self.assertEqual(allocation['physical_GPUs'],[0])
        self.assertEqual(allocation['identity'],self.hardware['identity'])
        self.assertEqual(allocation['source'],'resources/hardware.json')
        self.assertEqual(self.fingerprint(self.hardware_path),before)

    def test_foreign_or_changed_parallel_phase_is_not_attributed_to_this_run(self):
        before=self.fingerprint(self.hardware_path)
        for field in ('scientific_binding','protocol','model','source_sha256'):
            with self.subTest(field=field):
                self.entry(**{field:'0'*64})
                with self.assertRaises(ValueError):
                    resources.build(self.binding)
                self.assertFalse((self.root/'resource_ledger.json').exists())
                self.assertEqual(self.fingerprint(self.hardware_path),before)

    def refresh(self, assets, path_map=None):
        old=verified(self.root/'binding.json')
        mapping=old.get('path_map',{}) if path_map is None else path_map
        previous=old.get('asset_identity',canonical_digest(old.get('asset_config',{})))
        key=canonical_digest(dict(old=previous,new=assets,path_map=mapping))
        history=self.root/'history/asset_refresh'/key
        write(history/'binding.json',old)
        write(history/'protocol_before.json',verified(self.root/'protocol_lock.json'))
        value=old if path_map is None else dict(Resolver(mapping).rewrite(old),path_map=mapping)
        new=write(self.root/'binding.json',dict(value,asset_config=assets,asset_identity=canonical_digest(assets),input_identity=canonical_digest(assets)))
        write(self.root/'asset_refresh.json',dict(status='DONE',refresh_identity=key,
            old_binding=old['identity'],new_binding=new['identity'],history=str(history)))
        return new,history

    def relocated_binding(self):
        self.binding=write(self.root/'binding.json',dict(self.binding,
            asset_config={},asset_identity=canonical_digest({}),input_identity='original',path_map={},
            inherited=dict(inputs=['/TEST_ONLY/Replica/scene/frame.png'],
                           sibling='/TEST_ONLY/Replica2/scene/frame.png',
                           configuration=dict(gamma=.025),relative='Replica/scene/frame.png')))
        entry=self.entry()
        mapping={'/TEST_ONLY/Replica':'/TEST_ONLY/relocated/Replica'}
        binding,history=self.refresh({'external':{'fixture':'/TEST_ONLY/checkpoint'}},path_map=mapping)
        return entry,binding,history,mapping

    def test_declared_path_relocation_keeps_original_phase_and_literal_history(self):
        entry,binding,history,mapping=self.relocated_binding()
        before={path:self.fingerprint(path) for path in (self.hardware_path,self.entry_path,history/'binding.json')}
        result=resources.build(binding)
        carry=result['runtime_allocation']['binding_carry']
        self.assertEqual(carry['phase_binding'],entry['scientific_binding'])
        self.assertFalse(carry['non_asset_binding_fields_equal'])
        self.assertTrue(carry['non_asset_binding_fields_equal_after_declared_path_rewrite'])
        self.assertEqual(carry['asset_refresh_chain'][0]['declared_path_map'],mapping)
        self.assertEqual(verified(history/'binding.json')['inherited']['inputs'],['/TEST_ONLY/Replica/scene/frame.png'])
        self.assertEqual({path:self.fingerprint(path) for path in before},before)
        self.assertEqual(resources.build(binding)['identity'],result['identity'])
        # A subsequent asset-only refresh must preserve the earlier relocation.
        newer,_=self.refresh({'external':{'fixture':'/TEST_ONLY/new_checkpoint'}})
        carried=resources.build(newer)['runtime_allocation']['binding_carry']
        self.assertEqual(len(carried['asset_refresh_chain']),2)
        self.assertTrue(carried['non_asset_binding_fields_equal_after_declared_path_rewrite'])

    def test_declared_path_relocation_rejects_undeclared_payload_changes(self):
        from copy import deepcopy
        entry,binding,history,mapping=self.relocated_binding()
        for change in ('undeclared_path','numeric_setting','prefix_sibling'):
            with self.subTest(change=change):
                altered=deepcopy(binding)
                if change=='undeclared_path':
                    altered['inherited']['inputs']=['/TEST_ONLY/unrelated/frame.png']
                elif change=='numeric_setting':
                    altered['inherited']['configuration']['gamma']=.05
                else:
                    altered['inherited']['sibling']='/TEST_ONLY/relocated/Replica2/scene/frame.png'
                changed=write(self.root/'binding.json',altered)
                write(self.root/'asset_refresh.json',dict(verified(self.root/'asset_refresh.json'),new_binding=changed['identity']))
                with self.assertRaises(ValueError):
                    resources.build(changed)
                self.assertFalse((self.root/'resource_ledger.json').exists())

    def test_completed_external_refresh_chain_keeps_original_phase(self):
        self.binding=write(self.root/'binding.json',dict(self.binding,asset_config={},asset_identity=canonical_digest({}),input_identity='original'))
        entry=self.entry()
        old=self.fingerprint(self.entry_path)
        for assets in ({'external':{'Open3DIS':{'common_scoring_root':'/TEST_ONLY/first'}}},
                       {'external':{'Open3DIS':{'common_scoring_root':'/TEST_ONLY/second'}}}):
            binding,history=self.refresh(assets)
            ledger=resources.build(binding)
            phase=ledger['runtime_allocation']
            self.assertEqual(phase['identity'],entry['identity'])
            self.assertEqual(phase['physical_GPUs'],[0,2])
            self.assertEqual(phase['binding_carry']['phase_binding'],entry['scientific_binding'])
            self.assertEqual(phase['binding_carry']['current_binding'],binding['identity'])
            self.assertTrue(phase['binding_carry']['non_asset_binding_fields_equal'])
            before=self.fingerprint(self.root/'resource_ledger.json')
            self.assertEqual(resources.build(binding)['identity'],ledger['identity'])
            self.assertEqual(self.fingerprint(self.root/'resource_ledger.json'),before)
        self.assertEqual(self.fingerprint(self.entry_path),old)

    def test_global_protocol_refresh_requires_the_exact_original_scannet_scope(self):
        self.protocol=write(self.root/'protocol_lock.json',dict(
            scopes={'scannet200_val':{'scenes':['one'],'point_order':'original'}},
            official_sources={'evaluator_scan':{'sha256':'a'*64,'bytes':17}}))
        self.binding=write(self.root/'binding.json',dict(self.binding,asset_config={},asset_identity=canonical_digest({}),input_identity='original'))
        entry=self.entry()
        binding,history=self.refresh({'external':{'Open3DIS':{'common_scoring_root':'/TEST_ONLY/root'}}})
        current=write(self.root/'protocol_lock.json',dict(self.protocol,unrelated_Replica_metadata='updated'))
        result=resources.build(binding)
        self.assertEqual(result['runtime_allocation']['binding_carry']['phase_protocol'],entry['protocol'])
        self.assertEqual(result['runtime_allocation']['binding_carry']['current_protocol'],current['identity'])
        before=self.fingerprint(self.root/'resource_ledger.json')
        for change in ('point_order','evaluator'):
            changed={k:v for k,v in current.items()}
            if change=='point_order':
                changed['scopes']={'scannet200_val':{'scenes':['one'],'point_order':'different'}}
            else:
                changed['official_sources']={'evaluator_scan':{'sha256':'b'*64,'bytes':17}}
            write(self.root/'protocol_lock.json',changed)
            with self.assertRaises(ValueError): resources.build(binding)
            self.assertEqual(self.fingerprint(self.root/'resource_ledger.json'),before)

    def test_refresh_requires_linked_history_and_preserved_scientific_binding(self):
        self.binding=write(self.root/'binding.json',dict(self.binding,asset_config={},asset_identity=canonical_digest({}),input_identity='original'))
        self.entry()
        new,history=self.refresh({'external':{'Open3DIS':{'common_scoring_root':'/TEST_ONLY/root'}}})
        for change in ('gpu','link','snapshot'):
            with self.subTest(change=change):
                saved={p:p.read_bytes() for p in (self.root/'binding.json',self.root/'asset_refresh.json',history/'binding.json')}
                if change=='gpu':
                    changed=write(self.root/'binding.json',dict(new,gpu='2'))
                    write(self.root/'asset_refresh.json',dict(verified(self.root/'asset_refresh.json'),new_binding=changed['identity']))
                elif change=='link':
                    write(self.root/'asset_refresh.json',dict(verified(self.root/'asset_refresh.json'),old_binding='0'*64))
                else:
                    write(history/'binding.json',dict(self.binding,gpu='2'))
                with self.assertRaises(ValueError):
                    resources.build(verified(self.root/'binding.json'))
                self.assertFalse((self.root/'resource_ledger.json').exists())
                for path,raw in saved.items(): path.write_bytes(raw)


if __name__=='__main__':
    unittest.main(verbosity=2)
