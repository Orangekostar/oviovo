"""Independent H2/SAM2 confirmations run after map evaluation; preserve partial evidence."""
from pathlib import Path
import traceback
from .common import write,verified
from .transfer import require_transfer

def confirm(binding,fc,*,sam2_root=None,sam2_python=None):
    locked=require_transfer(binding);root=Path(binding['output_root'])
    from .holdout2 import run as h2
    from .real_proposals import run as real
    results={};errors={}
    for name,operation in [('H2',lambda:h2(binding,fc)),('REAL',lambda:real(binding,fc,sam2_root=sam2_root,sam2_python=sam2_python))]:
        try:results[name]=operation()
        except Exception as exc:traceback.print_exc();errors[name]=repr(exc)
    result=write(root/'confirmation/receipt.json',dict(status='PARTIAL_ACTIVE_STAGE' if errors else 'COMPLETE',
        predictor_lock=locked['identity'],stages={n:r['identity'] for n,r in results.items()},errors=errors,used_for_selection=False))
    if errors:raise RuntimeError('Active confirmation failed: '+repr(errors))
    return result
