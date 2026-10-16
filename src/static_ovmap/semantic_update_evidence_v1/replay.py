"""Reconstruct exactly the A-consumed tensors from the compact public bundle."""
from pathlib import Path
import numpy as np
from .common import verified,save_arrays,write

def restore_public_inputs(folder,output):
    folder=Path(folder);output=Path(output);meta=verified(folder/'A/import.json')
    with np.load(folder/'A/source_scores.npz',allow_pickle=False) as a:
        scores=a['full200_cosines'];defined=a['defined'];ids=a['valid_ids']
    embeddings=np.full((*scores.shape[:-1],768),np.nan,np.float32)
    with np.load(folder/'A/source_vectors.npz',allow_pickle=False) as a:
        if not np.array_equal(ids,a['valid_ids']):raise ValueError('Bundle vocabulary differs')
        columns=a['original_column_indices']
        for i,ci in enumerate(columns):embeddings[:,ci,0]=a['clean8V_embeddings'][:,i]
        text=a['text_embeddings']
    save_arrays(output/'A/import.npz',scores=scores,defined=defined,valid_ids=ids,embeddings=embeddings,text=text)
    (output/'A/import.json').write_bytes((folder/'A/import.json').read_bytes())
    # Only clean8V primary vectors were used by A. Auxiliary metrics use scores, not vectors.
    return dict(output_root=str(output),source_identity=meta['source_identity'],
                task_specification=__import__('json').loads((Path(__file__).resolve().parents[3]/'configs/static_ovmap/semantic_update_evidence_v1.json').read_text()))
