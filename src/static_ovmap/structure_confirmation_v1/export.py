"""Small-head exports and exact, lossless public prediction artifacts."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .common import verified,write,ConsumptionIndex,canonical_digest,FORWARD_KEYS,digest_file,STATICS,read_document
from .prediction_codec import encode,decode
from .evaluation import gzip_record

def pack_weights(source,dest,*,base_bytes=None):
    raw=Path(source).read_bytes();payload=np.frombuffer(raw,np.uint8).copy()
    if base_bytes is not None:
        n=min(len(payload),len(base_bytes));payload[:n]^=np.frombuffer(base_bytes,np.uint8)[:n]
    header=json.dumps(dict(format='EXACT_MODEL_ONLY_PT_BYTES_XOR_BASE_GZIP_V1',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),xor_base=base_bytes is not None),separators=(',',':')).encode()
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    with gzip.GzipFile(filename=str(dest),mode='wb',mtime=0,compresslevel=9) as stream:stream.write(len(header).to_bytes(4,'big'));stream.write(header);stream.write(payload.tobytes())
    return raw

def unpack_weights(path,*,base_bytes=None):
    with gzip.open(path,'rb') as stream:
        n=int.from_bytes(stream.read(4),'big');header=json.loads(stream.read(n));payload=np.frombuffer(stream.read(),np.uint8).copy()
    if header['format']!='EXACT_MODEL_ONLY_PT_BYTES_XOR_BASE_GZIP_V1' or len(payload)!=header['bytes'] or header['xor_base']!=(base_bytes is not None):raise ValueError('Weight transport metadata differs')
    if base_bytes is not None:
        n=min(len(payload),len(base_bytes));payload[:n]^=np.frombuffer(base_bytes,np.uint8)[:n]
    raw=payload.tobytes()
    if hashlib.sha256(raw).hexdigest()!=header['sha256']:raise ValueError('Lossless exported weight SHA differs')
    return raw

def load_export(folder,weight_key,*,device='cpu'):
    import torch
    from .models import make_models
    folder=Path(folder);manifest=verified(folder/'weights/manifest.json');row=manifest['weights'][weight_key]
    base=unpack_weights(folder/manifest['weights'][row['base']]['path']) if row['base'] else None
    raw=unpack_weights(folder/row['path'],base_bytes=base);state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    head=make_models(state['seed'])[state['method']];head.load_state_dict(state['model'],strict=True)
    return head.to(device).eval(),state

def export(binding,destination):
    root=Path(binding['output_root']);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);index=ConsumptionIndex();manifest=verified(root/'model_selection_lock.json');weights={};profiles={}
    from .models import make_models,static_model
    architecture_models={**make_models(17),**{name:static_model(name) for name in STATICS}}
    architecture=write(destination/'architecture.json',dict(status='COMPLETE',schema='FG2_STRUCTURE_CONFIRMATION_ARCHITECTURE_V1',specification=binding['task_specification'],predictor_lock=verified(root/'predictor_lock.json')['identity'],input_keys=sorted(FORWARD_KEYS),dimensions=dict(raw=1536,unit=768,metadata=8,maximum_views=8,maximum_sites=64),models={name:dict(module=str(head),parameters=sum(p.numel() for p in head.parameters()),trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),grouping=getattr(head,'grouping',None)) for name,head in architecture_models.items()},normalization='FC region and local_unit are parent-normalized; S/V/G/factorial pool raw then phi then L2; S_PRE applies phi to valid raw before pooling with no token L2; MV normalizes region tokens and final anchor-plus-residual',residual='FG variants use parent correct(base, group_vectors-base, alpha, eta); MV uses unit(base+eta*output_projection(pooled)); static means have no learned residual',permutation='External joint valid-token permutation; FG2_GROUP_CONTROL_V1 / key / condition / frame / salt9173; SHA256 first8 big endian; NumPy default_rng; prefix consistent',numerical_sources={name:digest_file(Path(__file__).with_name(name+'.py')) for name in ('models','grouping','data','training','recognition','checkpoints','selection')},model_initialization_seed_for_module_description=17))
    del architecture_models
    for arm in manifest['arms'].values():
        base_key=None;base_bytes=None
        for profile in ('matched1000','terminal2000','selected'):
            row=arm[profile];descriptor=row['model'];index.identity(descriptor['path'],descriptor);key=descriptor['sha256'];profiles[arm['method']+':seed'+str(arm['seed'])+':'+profile]=key
            if key not in weights:
                dest=destination/'weights'/(key+'.pt.gz');raw=pack_weights(descriptor['path'],dest,base_bytes=base_bytes)
                if unpack_weights(dest,base_bytes=base_bytes)!=raw:raise ValueError('Weight roundtrip failed')
                weights[key]=dict(path=str(dest.relative_to(destination)),source_sha256=key,source_bytes=len(raw),transport=index.identity(dest),base=base_key,method=arm['method'],seed=arm['seed'],step=row['step'],model_only=True)
            if profile=='matched1000':base_key=key;base_bytes=Path(descriptor['path']).read_bytes()
    write(destination/'weights/manifest.json',dict(status='COMPLETE',weights=weights,profiles=profiles,architecture=architecture['identity'],loader='static_ovmap.structure_confirmation_v1.export.load_export',optimizer_published=False,backbone_published=False))
    score_files=[]
    score_paths=sorted((root/'training').glob('seed*/*/dev_*.json'))+sorted((root/'recognition').glob('dev_B_*.json'))
    for p in score_paths:
        row=verified(p);score_files.append(dict(kind='DEV',source=p,arrays=row['arrays'],metadata={k:v for k,v in row.items() if k!='arrays'},array_name='full200_cosines'))
    for domain,lock_path in [('REAL',root/'confirmation/real_prediction_lock.json'),('GT',root/'confirmation/GT/prediction_lock.json')]:
        if not lock_path.exists():continue
        lock=verified(lock_path)
        for pointer in lock['records']:
            row=verified(pointer['path']);score_files.append(dict(kind=domain,source=Path(pointer['path']),arrays=row['arrays'],metadata={k:v for k,v in row.items() if k!='arrays'},array_name='full200_cosines'))
    for p in sorted((root/'map/readouts').glob('*/*.npz')):
        score_files.append(dict(kind='REG',source=p,arrays=index.identity(p),metadata={},array_name='cosines'))
    records=[]
    for row in score_files:
        index.identity(row['arrays']['path'],row['arrays'])
        with np.load(row['arrays']['path'],allow_pickle=False) as a:
            values=a[row['array_name']].copy();ids=a['valid_ids'].copy();metadata={k:a[k].tolist() for k in a.files if k in ('object_keys','class_ids','methods','slot_ids','predictions','defined')}
        # Map score arrays may be FP64 because original scene text is FP64. Preserve it exactly.
        key=canonical_digest(dict(kind=row['kind'],source=row['arrays']['sha256'],array_name=row['array_name']));dest=destination/'scores'/row['kind']/(key+('.scores.gz' if values.dtype==np.float32 else '.npz'))
        if values.dtype==np.float32:
            encode(dest,values)
            if decode(dest).tobytes()!=values.tobytes():raise ValueError('Published FP32 score roundtrip failed')
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,full_cosines=values)
            with np.load(dest,allow_pickle=False) as a:
                if a['full_cosines'].tobytes()!=values.tobytes():raise ValueError('Original FP64 map scores changed')
        records.append(dict(kind=row['kind'],source_arrays=row['arrays'],source_record=str(row['source']),path=str(dest.relative_to(destination)),dtype=str(values.dtype),shape=list(values.shape),valid_ids=ids.tolist(),metadata=metadata,record_metadata=row['metadata'],transport=index.identity(dest),lossless=True,embeddings_omitted=True))
    gzip_record(destination/'scores/manifest.json.gz',dict(status='COMPLETE',records=records,full_cosines_preserved=True,codec_source='src/static_ovmap/structure_confirmation_v1/prediction_codec.py'))
    # Derived result records and matcher PR statistics only; raw images/mesh/GT/optimizer excluded.
    records_out=[]
    patterns=('training/seed*/*/receipt.json','training/seed*/*/replay_*.json','training/seed*/*/processed_failure_*.json','training/seed*/draw_contract.json','training/workload_detail.json','recognition/dev_*.json','source_binding.json','class_split.json','split_manifest.json','model_selection_lock.json','predictor_lock.json','role_lock.json','roles.json','preflight.json','prepare_cache.json','workload.json','task_paths.json','confirmation/acquisition.json','confirmation/download_plan.json','confirmation/geometry_lock.json','confirmation/geometry/*/manifest.json','confirmation/SAM/generator_lock.json','confirmation/SAM/observed_exit.json','confirmation/real_input_lock.json','confirmation/features/manifest.json','confirmation/features/storage_estimate.json','confirmation/real_prediction_lock.json','confirmation/GT/input_lock.json','confirmation/GT/*/original_selection.json','confirmation/GT/*/manifest.json','confirmation/GT/prediction_lock.json','confirmation/GT/score.json','confirmation/real_score.json','confirmation/score.json','map/prediction_lock.json','map/score.json','statistics/**/*.json','cost/receipt.json','figures/*.json','engineering/*.json','execution/*.json','execution_observed.json','review/*.json')
    # Keep the exact pooled semantic confusion counts, as well as AP matches.
    patterns += ('map/pools/*/*_per_class.json',)
    for pattern in patterns:
        for source in sorted(root.glob(pattern)):
            relative=source.relative_to(root);dest=destination/'records'/relative.with_suffix('.json.gz')
            # These two metadata formats are intentionally unsealed. Their exact
            # source bytes remain bound by the outer artifact index.
            value=read_document(source,index,sealed=False) if relative==Path('task_paths.json') or source.name.endswith('verifications.json') else verified(source)
            gzip_record(dest,value);records_out.append(dict(source=index.identity(source),export=index.identity(dest)))
    for source in sorted((root/'confirmation').glob('pools/*/trace.json.gz'))+sorted((root/'confirmation').glob('pools/*/matches.json.gz')):
        relative=source.relative_to(root);dest=destination/'records'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());records_out.append(dict(source=index.identity(source),export=index.identity(dest)))
    map_score=root/'map/score.json'
    if map_score.exists():
        scored=verified(map_score);sufficient={};aliases=[]
        for row in scored['rows']:
            receipt=read_scoring_receipt(row['evaluation_receipt'])
            for kind in ('matches','trace'):
                source=Path(receipt['manifest']).with_name(kind+'.json.gz');descriptor=index.identity(source);key=descriptor['sha256']
                if key not in sufficient:
                    dest=destination/'REG_PR'/(key+'.json.gz');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
                    sufficient[key]=dict(kind=kind,path=str(dest.relative_to(destination)),source=descriptor,transport=index.identity(dest))
                aliases.append(dict(scene=row['scene'],method=row['method'],kind=kind,content_sha256=key))
        gzip_record(destination/'REG_PR/manifest.json.gz',dict(status='COMPLETE',blobs=sufficient,aliases=aliases,pooled_reproduction='Read each method in map/score.json ordered_scenes and namespace its per-scene matches before the unchanged released evaluate_matches',raw_GT_or_masks_published=False))
    for source in sorted((root/'statistics').glob('*/bootstrap.npz')):
        dest=destination/'records'/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());records_out.append(dict(source=index.identity(source),export=index.identity(dest)))
    for source in sorted((root/'training').glob('seed*/draws.npz')):
        dest=destination/'records'/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());records_out.append(dict(source=index.identity(source),export=index.identity(dest)))
    for source in sorted((root/'training').glob('seed*/*/updates.jsonl')):
        dest=destination/'records'/source.relative_to(root).with_suffix('.jsonl.gz');dest.parent.mkdir(parents=True,exist_ok=True)
        with gzip.GzipFile(filename=str(dest),mode='wb',mtime=0,compresslevel=9) as stream:stream.write(source.read_bytes())
        with gzip.open(dest,'rb') as stream:
            if stream.read()!=source.read_bytes():raise ValueError('Training ledger transport changed its bytes')
        records_out.append(dict(source=index.identity(source),export=index.identity(dest)))
    for source in sorted((root/'figures').rglob('*')):
        if source.suffix not in ('.png','.pdf','.svg'):continue
        dest=destination/'figures'/source.relative_to(root/'figures');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    files=[p for p in destination.rglob('*') if p.is_file() and p!=destination/'artifact_index.json'];sizes=[dict(path=str(p.relative_to(destination)),bytes=p.stat().st_size,sha256=index.identity(p)['sha256']) for p in files];total=sum(r['bytes'] for r in sizes)
    return write(destination/'artifact_index.json',dict(status='COMPLETE' if total<=256*2**20 and all(r['bytes']<=40*2**20 for r in sizes) else 'PARTIAL_PUBLICATION_SIZE',files=sizes,total_bytes=total,index_self_excluded=True,total_bytes_excludes_index=True,final_git_budget_check_includes_index_code_and_docs=True,maximum_blob_bytes=max((r['bytes'] for r in sizes),default=0),weights=len(weights),weight_profiles=profiles,score_records=len(records),derived_records=records_out,raw_source_data_published=False,optimizer_published=False))

def read_scoring_receipt(path):
    from .common import read_document
    return read_document(path,sealed=False)
