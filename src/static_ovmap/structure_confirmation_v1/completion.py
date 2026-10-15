"""Validate completed training without replacing its measured execution receipt."""
from pathlib import Path
from .common import verified,ConsumptionIndex,unit_key,METHODS,CHECKPOINTS

def completed_training(binding):
    # Keep CPU orchestration independent of the FC-only timm training dependency.
    producers=('common','models','grouping','data','recognition','checkpoints','selection','training')
    root=Path(binding['output_root']);path=root/'training/receipt.json'
    if not path.exists():return None
    value=verified(path);prepare=verified(root/'prepare_cache.json');features=verified(root/'features/train-dev.json')
    expected={(method,seed) for method in METHODS for seed in (17,29)}
    actual={(row['method'],row['seed']) for row in value['arms']}
    if value['status']!='COMPLETE' or actual!=expected or len(value['arms'])!=16 or value['nominal_updates']!=32000 or value['frozen_before']!=value['frozen_after']:
        raise ValueError('Completed training receipt has incomplete coverage')
    index=ConsumptionIndex(root/'training/completion_verifications.json')
    for row in value['arms']:
        method,seed=row['method'],row['seed'];arm=verified(root/f'training/seed{seed}/{method}/receipt.json');draws=verified(root/f'training/seed{seed}/draw_contract.json')
        identity=unit_key('train_arm',dict(binding=binding['identity'],prepare=prepare['identity'],features=features['identity'],FC=prepare['FC_identity'],draws=draws['draw_sha']),producers=producers,seed=seed,method=method)
        if arm['identity']!=row['identity'] or arm['input_identity']!=identity or arm['status']!='COMPLETE' or arm['completed_steps']!=2000 or [r['step'] for r in arm['validations']]!=list(CHECKPOINTS):
            raise ValueError('Completed arm producer/input/checkpoint identity differs: '+method+':'+str(seed))
        for candidate in arm['validations']:
            index.identity(candidate['model']['path'],candidate['model']);dev=verified(candidate['path'])
            if dev['identity']!=candidate['recognition']:raise ValueError('Completed DEV receipt changed')
            index.identity(dev['arrays']['path'],dev['arrays'])
    index.write_memo(root/'training/completion_verifications.json')
    return value
