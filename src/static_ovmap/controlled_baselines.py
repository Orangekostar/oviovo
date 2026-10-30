"""Original baseline inference on a predeclared paper-sized, GT-blind subset."""
import argparse
import ast
import contextlib
import hashlib
import os
from pathlib import Path
import runpy
import sys
import types

import numpy as np
from static_ovmap.benchmark_completion_v2.common import verified,write,digest_file,read,canonical_digest


def controlled_scenes(cohorts,validation):
    replica=cohorts['replica8'];paper=cohorts['scannet_cf18']
    if len(set(replica))!=len(replica) or len(set(paper))!=len(paper):
        raise ValueError('Duplicate predeclared scene')
    allowed=set(validation)
    return dict(replica48=list(replica),scannet200_val=[s for s in paper if s in allowed]),[s for s in paper if s not in allowed]


def frame_mapping(frames):
    if len(frames)!=200 or frames!=sorted(set(frames)) or any(type(i) is not int or i<0 for i in frames):
        raise ValueError('Exactly200 predeclared source frames in time order are required')
    return [dict(source=frame,staged=i*10) for i,frame in enumerate(frames)]


def object_vocabulary(taxonomy):
    mapping=dict(zip(taxonomy['text_ids'],taxonomy['text_names']))
    ids=taxonomy['instance_ids']
    if not set(ids)<=set(mapping):raise ValueError('Object vocabulary is not on the fixed text axis')
    return list(ids),[mapping[i] for i in ids]


def load_frozen_checkpoint(model,state):
    import torch
    expected=model.state_dict();missing=sorted(set(expected)-set(state))
    assert missing in [[],['criterion.empty_weight']] and not set(state)-set(expected)
    complete=dict(state)
    if missing:complete['criterion.empty_weight']=expected['criterion.empty_weight']
    model.load_state_dict(complete,strict=True)
    assert all(torch.equal(model.state_dict()[key].cpu(),value.cpu()) for key,value in state.items())
    return missing


def descriptor(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=digest_file(path))


def check(d):
    assert Path(d['path']).stat().st_size==d['bytes'] and digest_file(d['path'])==d['sha256'],d['path']


def link(target,source):
    target,source=Path(target),Path(source)
    if target.is_symlink():
        if target.resolve()!=source.resolve():raise ValueError('Changed staged input: '+str(target))
    elif target.exists():raise ValueError('Untracked file occupies staged input: '+str(target))
    else:
        target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source.resolve())


def prepare(root,out,cohort_path,source_root):
    assert not (out/'scope.json').exists()
    protocol=verified(root/'protocol_lock.json');cohorts=read(cohort_path)['cohorts']
    scopes,excluded=controlled_scenes(cohorts,protocol['scopes']['scannet200_val']['scenes'])
    assert len(scopes['replica48'])==8 and len(scopes['scannet200_val'])==15
    scenes={}
    for scene in scopes['scannet200_val']:
        frames=verified(root/'benchmark/frames/scannet200_val'/scene/'frame_plan.json')
        mapping=frame_mapping(frames['frame_ids']);folder=out/'rgbd'/scene
        for item in frames['outputs']:check(item)
        for item in mapping:
            for sub,ext in [('color','.jpg'),('depth','.png'),('pose','.txt')]:
                source=root/'benchmark/frames/scannet200_val'/scene/sub/(str(item['source'])+ext)
                link(folder/sub/(str(item['staged'])+ext),source)
        for name in ('intrinsic_color','intrinsic_depth','extrinsic_color','extrinsic_depth'):
            link(folder/'intrinsic'/(name+'.txt'),root/'benchmark/frames/scannet200_val'/scene/'intrinsic'/(name+'.txt'))
        raw=verified(root/'assets/datasets/scannet/receipts'/(scene+'.json'))['files']
        mesh=Path(raw['_vh_clean_2.ply']['path']);segments=mesh.with_name(scene+'_vh_clean_2.0.010000.segs.json')
        link(folder/mesh.name,mesh)
        scenes[scene]=dict(frame_plan=descriptor(root/'benchmark/frames/scannet200_val'/scene/'frame_plan.json'),
            frozen_frame_outputs=frames['outputs'],mapping=mapping,mesh=descriptor(mesh),segments=descriptor(segments),
            rgbd=str(folder),GT_labels_read=False)
    weights=dict(mask3d=out.parent/'assets/scannet200_val.ckpt',sam=root/'assets/weights/OpenMask3D/sam_vit_h_4b8939.pth',
        clip_om=root/'assets/weights/OpenMask3D/ViT-L-14-336px.pt',
        clip_mc=root/'assets/weights/MaskClustering/open_clip_pytorch_model.bin',
        cropformer=root/'assets/acquisition_20261026/Mask2Former_hornet_3x_576d0b.pth')
    from static_ovmap.benchmark_completion_v2.external import PINS
    sources={}
    for method in ['OpenMask3D','MaskClustering']:
        import subprocess
        repo=root/'assets/external'/method;pin=PINS[method][1]
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==pin
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',pin],cwd=repo,text=True).splitlines()
        for name in names:
            if Path(name).suffix not in ('.py','.yaml','.json','.sh'):continue
            original=subprocess.check_output(['git','show',pin+':'+name],cwd=repo)
            path=repo/name;assert digest_file(path)==hashlib.sha256(original).hexdigest(),str(path)
            sources[str(path)]=digest_file(path)
    write(out/'scope.json',dict(status='DONE',scopes=scopes,excluded_paper_training_scenes=excluded,
        cohort_source=descriptor(cohort_path),paper='https://arxiv.org/html/2603.26541v1#S11',
        selection='FIXED_PAPER_CF18_INTERSECTION_WITH_FROZEN_VALIDATION_SET;NO_RESULT_BASED_SELECTION',
        scenes=scenes,weights={k:descriptor(p) for k,p in weights.items()},sources=sources,
        adapter=descriptor(source_root/'src/static_ovmap/controlled_baselines.py'),
        taxonomy={b:{k:protocol['scopes'][b][k] for k in ['text_ids','text_names','instance_ids']} for b in scopes},
        inference_scope='ONLY_TWO_MISSING_SCANNET_BASELINES_15_SCENES',views=200,
        frame_adapter='ORIGINAL_SELECTED_FRAME_BYTES;STAGED_FRAME_ID=10*SELECTED_INDEX',
        native_51_200_semantic_benchmark=False,point_metric_scope='COMMON_AP_OBJECT_CLASSES_48_198',
        imported_Open3DIS_ScanNet_view_budget='AUTHOR_ARCHIVE;NOT_PROVEN_MATCHED_TO_200_FRAMES',
        old_GT_exposure_erased=False,new_training_updates=0))
    print('CONTROLLED_SCOPE_FROZEN',scopes,flush=True)


def source_check(scope,method):
    for path,sha in scope['sources'].items():
        if '/external/'+method+'/' in path:assert digest_file(path)==sha,path


def maskclustering(root,out,scene,stage,scope):
    from static_ovmap.benchmark_completion_v2.mask_clustering_full import _author_context,_local_clip
    repo=root/'assets/external/MaskClustering';work=out/'MaskClustering';source_check(scope,'MaskClustering')
    args=argparse.Namespace(**dict(read(repo/'configs/scannet.json'),config='scannet_controlled',
        seq_name=scene,seq_name_list=scene,debug=False))
    folder=Path(scope['scenes'][scene]['rgbd']);link(work/'data/scannet/processed'/scene,folder)
    text=work/'data/text_features/scannet.npy'
    class_ids,class_names=object_vocabulary(scope['taxonomy']['scannet200_val'])
    def factory(request):
        assert request.seq_name==scene
        from dataset.scannet import ScanNetDataset
        dataset=ScanNetDataset(scene)
        dataset.get_label_features=lambda:np.load(text,allow_pickle=True).item()
        ids=dataset.get_label_id()[0]
        assert list(ids)==class_names
        assert list(ids.values())==class_ids
        return dataset
    with _author_context(repo,work,args,factory):
        if stage in ['cluster','query']:
            runpy.run_path(str(repo/('main.py' if stage=='cluster' else 'semantics/open-voc_query.py')),run_name='__main__')
        elif stage=='frontend':
            project=Path('/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/Entity/Entityv2/CropFormer')
            sys.path[:0]=[str(project),str(project/'demo_cropformer')]
            sys.argv=[str(repo/'mask_predict.py'),'--config-file',str(project/'configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml'),
                '--root',str(out/'rgbd'),'--seq_name_list',scene,'--image_path_pattern','color/*0.jpg','--dataset','scannet',
                '--opts','MODEL.WEIGHTS',scope['weights']['cropformer']['path']]
            runpy.run_path(str(repo/'mask_predict.py'),run_name='__main__')
        elif stage in ['features','text']:
            with _local_clip(scope['weights']['clip_mc']['path']):
                if stage=='features':
                    runpy.run_path(str(repo/'semantics/get_open-voc_features.py'),run_name='__main__')
                else:
                    import torch,open_clip
                    from open_clip import tokenizer
                    source=repo/'semantics/extract_label_featrues.py'
                    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['load_clip','extract_text_feature']]
                    namespace=dict(np=np,torch=torch,open_clip=open_clip,tokenizer=tokenizer)
                    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
                    namespace['model']=namespace['load_clip']();text.parent.mkdir(parents=True,exist_ok=True)
                    namespace['extract_text_feature'](str(text),class_names)
        else:raise ValueError(stage)


def openmask(root,out,scene,stage,scope):
    import torch,yaml
    from static_ovmap.benchmark_completion_v2.openmask_full import _author_context,_local_clip
    from static_ovmap.benchmark_completion_v2.openmask_io import _query_class
    repo=root/'assets/external/OpenMask3D';work=out/'OpenMask3D'/scene;work.mkdir(parents=True,exist_ok=True)
    source_check(scope,'OpenMask3D');data=scope['scenes'][scene];folder=Path(data['rgbd'])
    masks=work/(scene+'_masks.pt');features=work/(scene+'_openmask3d_features.npy');text=out/'OpenMask3D/text.npy'
    binding=dict(output_root=str(root),asset_config={})
    with _author_context(repo,[str(__file__)]):
        if stage=='masks':
            import hydra
            from hydra.experimental import compose,initialize_config_dir
            from trainer.trainer import InstanceSegmentation
            from utils.point_cloud_utils import load_ply_with_normals
            from datasets.scannet200.scannet200_constants import VALID_CLASS_IDS_200,CLASS_LABELS_200,SCANNET_COLOR_MAP_200
            source=repo/'openmask3d/class_agnostic_mask_computation/conf'
            processed=work/'processed';processed.mkdir(exist_ok=True)
            coords,colors_normals,_=load_ply_with_normals(data['mesh']['path'])
            segments=np.asarray(read(data['segments']['path'])['segIndices'])
            assert len(coords)==len(segments)
            segments=np.unique(segments,return_inverse=True)[1]
            np.save(processed/'points.npy',np.hstack((coords,colors_normals,segments[:,None])).astype(np.float32))
            (processed/'test_database.yaml').write_text(yaml.safe_dump([dict(filepath=str(processed/'points.npy'),raw_filepath=data['mesh']['path'])]))
            labels={int(i):dict(color=list(SCANNET_COLOR_MAP_200[i]),name=n,validation=True) for i,n in zip(VALID_CLASS_IDS_200,CLASS_LABELS_200)}
            (processed/'label_database.yaml').write_text(yaml.safe_dump(labels))
            overrides=['model.num_queries=150','data.test_mode=test','data.train_mode=test','data.validation_mode=test',
                'general.mask_save_dir='+str(work),'general.save_visualizations=false']
            for name in ['train_dataset','test_dataset','validation_dataset']:
                overrides += ['data.'+name+'.data_dir='+str(processed),
                    'data.'+name+'.label_db_filepath='+str(processed/'label_database.yaml')]
            with initialize_config_dir(config_dir=str(source),job_name='controlled_scannet'):
                cfg=compose(config_name='config_base_class_agn_masks_scannet200.yaml',overrides=overrides)
            dataset=hydra.utils.instantiate(cfg.data.test_dataset);batch=hydra.utils.instantiate(cfg.data.test_collation)([dataset[0]])
            model=InstanceSegmentation(cfg);checkpoint=torch.load(scope['weights']['mask3d']['path'],map_location='cpu')
            state=checkpoint.get('state_dict',checkpoint);missing=load_frozen_checkpoint(model,state)
            assert cfg.data.test_mode=='test'
            write(work/'model_compatibility.json',dict(status='DONE',checkpoint=scope['weights']['mask3d'],
                missing_keys=missing,all_checkpoint_tensors_loaded_bitwise=True,all_inference_model_tensors_loaded=True,
                initialized_training_loss_buffer_only=bool(missing),training_loss_called=False,test_mode=cfg.data.test_mode))
            model.validation_dataset=dataset;model.test_dataset=dataset;model.cuda().eval()
            batch=model.transfer_batch_to_device(batch,torch.device('cuda:0'),0)
            with torch.no_grad():model.eval_step(batch,0,input_mode='full_dataset')
            assert masks.is_file()
        elif stage=='features':
            from openmask3d.data.load import Camera,Images,InstanceMasks3D,PointCloud
            from openmask3d.mask_features_computation.features_extractor import FeaturesExtractor
            from PIL import Image
            config=yaml.safe_load((repo/'openmask3d/configs/openmask3d_inference.yaml').read_text())
            width,height=Image.open(folder/'color/0.jpg').size
            indices=[item['staged'] for item in data['mapping']]
            camera=Camera(str(folder/'intrinsic/intrinsic_color.txt'),[height,width],str(folder/'pose'),str(folder/'depth'),'.png',1000.)
            with _local_clip(scope['weights']['clip_om']['path']):
                extractor=FeaturesExtractor(camera=camera,clip_model=config['external']['clip_model'],
                    images=Images(str(folder/'color'),'.jpg',indices),masks=InstanceMasks3D(str(masks)),
                    pointcloud=PointCloud(data['mesh']['path']),sam_model_type=config['external']['sam_model_type'],
                    sam_checkpoint=scope['weights']['sam']['path'],vis_threshold=config['openmask3d']['vis_threshold'],device=torch.device('cuda:0'))
                params=config['openmask3d'];values=extractor.extract_features(topk=params['top_k'],
                    multi_level_expansion_ratio=params['multi_level_expansion_ratio'],num_levels=params['num_of_levels'],
                    num_random_rounds=params['num_random_rounds'],num_selected_points=params['num_selected_points'],
                    save_crops=False,out_folder=str(work),optimize_gpu_usage=False)
            np.save(features,values)
        elif stage=='text':
            import clip
            with _local_clip(scope['weights']['clip_om']['path']):model,_=clip.load('ViT-L/14@336px',device='cuda:0')
            cls=_query_class(binding,with_clip=True);value=cls.__new__(cls)
            value.device='cuda:0';value.clip_model=model;value.feature_size=768
            value.query_sentences=['a {} in a scene'.format(n) for n in scope['taxonomy']['scannet200_val']['text_names']]
            np.save(text,value.get_text_query_embeddings().numpy())
        elif stage=='query':
            cls=_query_class(binding);value=cls.__new__(cls);value.text_query_embeddings=np.load(text)
            value.label_mapper=np.vectorize(dict(enumerate(scope['taxonomy']['scannet200_val']['text_ids'])).get)
            with np.errstate(divide='ignore',invalid='ignore'):
                masks_,classes,scores=value.compute_classes_per_mask_diff_scores(str(masks),str(features),keep_first=None)
            np.savez_compressed(work/'prediction.npz',pred_masks=np.asarray(masks_),pred_classes=classes,pred_scores=scores)
        else:raise ValueError(stage)


def worker(root,out,method,scene,stage,scope_path=None):
    scope=verified(scope_path or out/'scope.json');assert scene in scope['scopes']['scannet200_val']
    for item in scope['weights'].values():check(item)
    for item in scope['scenes'][scene]['frozen_frame_outputs']:check(item)
    for k in ['mesh','segments','frame_plan']:check(scope['scenes'][scene][k])
    assert digest_file(__file__)==scope['adapter']['sha256']
    (maskclustering if method=='MaskClustering' else openmask)(root,out,scene,stage,scope)
    print('CONTROLLED_ORIGINAL_STAGE_DONE',method,scene,stage,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=['OpenMask3D','MaskClustering']);p.add_argument('--scene');p.add_argument('--stage')
    p.add_argument('--scope-file',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--source-root',type=Path);p.add_argument('--cohorts',type=Path)
    args=p.parse_args()
    if args.prepare:prepare(args.root,args.out,args.cohorts,args.source_root)
    else:worker(args.root,args.out,args.method,args.scene,args.stage,args.scope_file)

if __name__=='__main__':main()
