"""Keyed, verified original NPZ import; no backbone or RGB loads."""
from pathlib import Path
import numpy as np
from .common import (REPO, METHODS, read, verified, write, ConsumptionIndex,
                     PathResolver, DependencyError, save_arrays,save_memo)

def public_GT_scores(folder,key,index=None):
    """Exact score-only fallback, joined on record_metadata.key, never REG."""
    from static_ovmap.structure_confirmation_v1.prediction_codec import decode
    folder=Path(folder);manifest=folder/'scores/manifest.json.gz'
    if index is not None:index.identity(manifest)
    rows=[r for r in read(manifest)['records'] if r['kind']=='GT' and r['record_metadata']['key']==key]
    if len(rows)!=1:raise DependencyError('Public GT score key missing or ambiguous: '+key)
    row=rows[0]
    if row['dtype']!='float32' or not row['lossless']:raise ValueError('Expected exact FP32 GT codec')
    path=folder/row['path']
    if index is not None:index.identity(path,row['transport'])
    scores=decode(path);defined=np.asarray(row['metadata']['defined'],bool);ids=np.asarray(row['valid_ids'],np.int64)
    if scores.shape[:-1]!=defined.shape or scores.shape[-1]!=len(ids):raise ValueError('Public score alignment differs')
    return scores,defined,ids

def column_index(columns, method, seed, *, profile='matched1000', prefix=8):
    rows = [i for i, c in enumerate(columns) if c['method'] == method and c['prefix'] == prefix
            and c['profile'] == ('selected' if method == 'B_FC' else profile)
            and c['seed'] == (None if method == 'B_FC' else seed)]
    if len(rows) != 1:
        raise DependencyError(f'Missing/ambiguous column {method}/{seed}/{profile}/{prefix}')
    c = columns[rows[0]]
    if profile == 'matched1000' and method != 'B_FC' and c['step'] != 1000:
        raise ValueError('Matched1000 is not step1000')
    return rows[0]

def import_originals(binding):
    root = Path(binding['output_root']); parent = Path(binding['STRUCT_ROOT'])
    index = ConsumptionIndex(root / 'input_verifications.json'); resolver = PathResolver(binding['path_map'])
    def document(path, expected=None):
        path = Path(resolver.resolve(str(path))); index.identity(path, expected)
        return resolver.rewrite(verified(path))
    inputs = document(parent / 'confirmation/GT/input_lock.json')
    lock = document(parent / 'confirmation/GT/prediction_lock.json')
    split = document(parent / 'class_split.json')
    objects = inputs['objects']; keys = [o['key'] for o in objects]
    records = {r['key']: r for r in lock['records']}
    if len(set(keys)) != len(keys) or set(keys) != set(records) or len(records) != len(lock['records']):
        raise ValueError('Original key domain differs from prediction lock')
    keep = [i for i, c in enumerate(lock['columns']) if c['method'] in METHODS]
    columns = [lock['columns'][i] for i in keep]
    scores=[]; vectors=[]; defined=[]; ids=None; used=[]
    for key in keys:
        row = records[key]; path = Path(resolver.resolve(row['path']))
        meta = document(path)
        if meta['key'] != key or meta['identity'] != row['identity']:
            raise ValueError('Object metadata/key identity changed')
        npz = Path(resolver.resolve(meta['arrays']['path']))
        if not npz.is_file():
            # Public data have exact full200 scores, but explicitly omit vectors.
            public,public_defined,public_ids=public_GT_scores(REPO/'artifacts/static_ovmap/structure_confirmation_v1',key,index)
            from hashlib import sha256
            save_arrays(root/'A/partial_scores'/(sha256(key.encode()).hexdigest()+'.npz'),
                        full200_cosines=public,defined=public_defined,valid_ids=public_ids)
            raise DependencyError('Original vectors missing; exact public scores recovered; bounded clean8V vector restoration required: '+str(npz))
        index.identity(npz, meta['arrays'])
        with np.load(npz, allow_pickle=False) as a:
            if ids is None: ids = a['valid_ids'].copy()
            if not np.array_equal(ids, a['valid_ids']): raise ValueError('Vocabulary row order changed')
            s=a['full200_cosines'][keep]; z=a['embeddings'][keep]; d=a['defined'][keep]
            if s.shape != (len(keep),4,200) or z.shape != (len(keep),4,768): raise ValueError('Prediction shape differs')
            if not np.isfinite(s[d]).all() or not np.isfinite(z[d]).all(): raise ValueError('Defined output nonfinite')
            scores.append(s); vectors.append(z); defined.append(d)
        used.append(dict(key=key, original_identity=row['identity'], arrays=meta['arrays']))
    context = binding['inherited']['scenes'][binding['inherited']['cohorts']['scannet_cf18'][0]]['context']
    ctx = document(context['path'], context); desc=ctx['FC_text']; path=Path(resolver.resolve(desc['path']))
    index.identity(path, desc)
    with np.load(path, allow_pickle=False) as a:
        text=a['text_embeddings'].astype(np.float32)
        if not np.array_equal(text.astype(a['text_embeddings'].dtype),a['text_embeddings']): raise ValueError('Text restoration lossy')
        if not np.array_equal(ids,a['valid_ids']): raise ValueError('Text row order differs')
    s=np.stack(scores);z=np.stack(vectors);d=np.stack(defined)
    if not np.allclose(np.linalg.norm(z[d],axis=-1),1.,atol=1e-5,rtol=0): raise ValueError('Original embeddings are not unit vectors')
    if not np.allclose(np.linalg.norm(text,axis=-1),1.,atol=1e-4,rtol=0): raise ValueError('Frozen text not unit')
    # CAP recomputes FP32 CPU scores with the original text; preserve stored source scores separately.
    err=float(np.max(np.abs((z[d] @ text.T)-s[d])))
    if err>1e-5: raise ValueError('Original embedding/text scores disagree: '+str(err))
    save_arrays(root/'A/import.npz',scores=s, embeddings=z, defined=d, valid_ids=ids, text=text)
    evidence=write(root/'A/import.json',dict(status='COMPLETE',source_identity=binding['source_identity'],
          input_lock=inputs['identity'],prediction_lock=lock['identity'],objects=[{k:o[k] for k in ('key','scene','family','class_id','base','supported')} for o in objects],
          columns=columns, original_objects=len(objects),base_objects=sum(o['base'] for o in objects),
          novel_objects=sum(not o['base'] for o in objects),supported_originals=sum(o['supported'] for o in objects),
          abstention_originals=sum(not o['supported'] for o in objects),class_split=split,text=desc,
          original_prediction_files=used,embedding_source='ORIGINAL_OUT_NPZ',new_head_calls=0,
          original_score_reconstruction_max_absolute_error=err,text_renormalized=False))
    save_memo(index,root/'input_verifications.json')
    return evidence
