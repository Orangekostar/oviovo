"""Fixed DEV rule; repeat cannot replace the seed17 nominated architecture."""
from static_ovmap.preservation_routing.selection import summarize,qualifies,compare,choose,TOL
from .common import verified,write
from pathlib import Path

def path_counts(main_nominee,same_repeat_qualified):
    return (8000,0,0) if main_nominee is None else (16000,286,22) if same_repeat_qualified else (16000,0,0)

def nominate(binding,seed):
    root=Path(binding['output_root']);receipt=verified(root/f'training/seed{seed}/receipt.json')
    rows=[verified(root/f'training/seed{seed}/{m}/receipt.json') for m in receipt['arms']]
    candidates=[dict(r['selected_metrics'],branch=r['branch'],order=i,checkpoint=r['selected_checkpoint'])
                for i,r in enumerate(rows) if r['qualified']]
    winner=choose(candidates);fixed=winner['branch'] if winner else None
    if seed==29:fixed=verified(root/'nomination_seed17.json')['nominee']
    repeated=bool(seed==29 and fixed is not None and any(r['branch']==fixed and r['qualified'] for r in rows))
    return write(root/f'nomination_seed{seed}.json',dict(status='COMPLETE',seed=seed,nominee=fixed,winner=winner,
        same_nominee_qualified=repeated,any_qualified=bool(candidates),
        checkpoints={r['branch']:r['selected_checkpoint'] for r in rows},arms={r['branch']:r['selected_metrics'] for r in rows},
        qualified={r['branch']:r['qualified'] for r in rows},selection_only_DEV_base=True,selection_uses_runtime=False))
