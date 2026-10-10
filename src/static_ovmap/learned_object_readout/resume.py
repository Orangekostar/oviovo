"""Finish an owed DEV check at the saved weights without advancing training."""
from contextlib import contextmanager
import math
from pathlib import Path
import shutil
import time

from .common import ConsumptionIndex, canonical_digest, verified, write

DEV_STEPS = (500, 1000, 1500, 2000)


def validate_branch_receipt(receipt):
    if receipt['status'] != 'COMPLETE' or receipt['completed_steps'] != 2000:
        raise ValueError('Complete fixed2000-update branch required')
    if receipt['branch'] == 'WARMUP':
        return
    rows = receipt['validations']
    if sorted(r['step'] for r in rows) != list(DEV_STEPS):
        raise ValueError('All four fixed DEV validation receipts are required')
    winner = None
    for row in sorted(rows, key=lambda r: r['step']):
        actual = verified(row['receipt'])
        if (actual['step'] != row['step'] or actual['seed'] != receipt['seed']
                or actual['branch'] != receipt['branch'] or actual['original_objects'] < 1
                or actual['classification_denominator'] != '160_BASE_CLASSES'):
            raise ValueError('Actual DEV validation scope changed')
        if not math.isfinite(row['criterion']) or actual['criterion'] != row['criterion']:
            raise ValueError('Actual DEV criterion differs from checkpoint selection')
        if winner is None or row['criterion'] < winner['criterion']-1e-12:
            winner = row
    if receipt['selected_step'] != winner['step'] or receipt['DEV_criterion'] != winner['criterion']:
        raise ValueError('Fixed DEV criterion/earlier-step selection changed')


def verify_completed_seed(binding, seed):
    root = Path(binding['output_root'])
    summary = verified(root/'training'/f'seed{seed}'/'receipt.json')
    if summary['status'] != 'COMPLETE':
        raise ValueError('Nomination requires complete scientific training')
    index = ConsumptionIndex(root/'training/verifications.json')
    for name in ('WARMUP', *summary['branches']):
        row = verified(root/'training'/f'seed{seed}'/name/'receipt.json')
        validate_branch_receipt(row)
        if name != 'WARMUP' and row['identity'] != summary['branches'][name]:
            raise ValueError('Seed summary and actual branch identity differ')
        for key in ('last_checkpoint', 'selected_checkpoint'):
            if row.get(key):
                index.identity(row[key]['path'], row[key])
    return summary


def repair_validation(directory, input_identity, fc, loader, dev_objects):
    import torch
    from . import training
    from .model import ReadoutHead
    directory = Path(directory); last = directory/'last.pt'
    state = torch.load(last, map_location='cpu', weights_only=False)
    step, branch = state['step'], state['branch']
    if (state['input_identity'] != input_identity or state['sampler_index'] != step*16
            or step < 0 or step > 2000 or state['seed'] not in (17, 29)):
        raise ValueError('Checkpoint identity or fixed sampler position changed')
    if branch == 'WARMUP':
        return None
    if branch not in training.BRANCHES:
        raise ValueError('Unknown fixed scientific branch')
    validations = list(state.get('validations', []))
    recorded = [v['step'] for v in validations]
    if len(set(recorded)) != len(recorded) or any(v not in DEV_STEPS or v > step for v in recorded):
        raise ValueError('Checkpoint DEV validation steps changed')
    missing = [v for v in DEV_STEPS if v <= step and v not in recorded]
    if not missing:
        return None
    if any(v < step for v in missing):
        raise ValueError('Missing historical DEV weights cannot be recovered from a later checkpoint')
    receipt_path = directory/'receipt.json'
    if receipt_path.exists():
        previous = verified(receipt_path)
        if previous['input_identity'] != input_identity or previous['status'] == 'COMPLETE':
            raise ValueError('Completed or differently bound branch cannot be rewritten')
    started = time.perf_counter()
    head = ReadoutHead(training.BRANCHES[branch][0]).to(fc.device)
    optimizer = training.optimizer_for(head)
    training.restore_checkpoint(last, head, optimizer, input_identity=input_identity)
    before = training.model_digest(head)
    validation = training.evaluate_dev(head, loader, dev_objects, fc)
    if training.model_digest(head) != before:
        raise ValueError('DEV validation unexpectedly changed learned weights')
    # Construction/evaluation must not consume any saved phase RNG state.
    training.restore_checkpoint(last, head, optimizer, input_identity=input_identity)
    validation.update(step=step, seed=state['seed'], branch=branch)
    path = directory/f'dev_{step:04d}.json'
    write(path, validation)
    validations.append(dict(step=step, criterion=validation['criterion'], receipt=str(path)))
    winner = training.select_earliest(validations)
    selected = directory/'selected.pt'
    if winner['step'] != step:
        selected_state = torch.load(selected, map_location='cpu', weights_only=False)
        if (selected_state['input_identity'] != input_identity or selected_state['step'] != winner['step']
                or selected_state['criterion'] != winner['criterion']):
            raise ValueError('Earlier selected DEV checkpoint is missing or changed')
    index = ConsumptionIndex(); original = index.identity(last)
    backup = directory/f'last_before_dev_{step:04d}_{time.time_ns()}.pt'
    shutil.copyfile(last, backup)
    if winner['step'] == step:
        training.save_checkpoint(selected, head, optimizer, seed=state['seed'], branch=branch, step=step,
                                 input_identity=input_identity, criterion=winner['criterion'], validations=validations)
    omitted = {'model','optimizer','seed','branch','step','sampler_index','input_identity','scheduler','RNG'}
    metadata = {k:v for k,v in state.items() if k not in omitted}
    metadata.update(validations=validations, elapsed_seconds=state.get('elapsed_seconds',0.)+time.perf_counter()-started)
    training.save_checkpoint(last, head, optimizer, seed=state['seed'], branch=branch, step=step,
                             input_identity=input_identity, **metadata)
    return write(directory/f'dev_resume_{step:04d}.json',dict(status='COMPLETE',seed=state['seed'],branch=branch,
                 input_identity=input_identity,completed_validation_step=step,optimizer_updates=0,
                 before_checkpoint=original,preserved_checkpoint=index.identity(backup),after_checkpoint=index.identity(last),
                 producer=index.identity(__file__),model_digest_before=before,model_digest_after=training.model_digest(head),
                 elapsed_seconds=time.perf_counter()-started,holdout_annotation_reads=0))


@contextmanager
def validation_resume_guard():
    from . import training
    original = training.train_phase
    def guarded(binding, fc, loader, train_objects, dev_objects, schedule, *, seed, branch, warmup=None):
        root = Path(binding['output_root']); directory = root/'training'/f'seed{seed}'/branch
        if branch != 'WARMUP' and (directory/'last.pt').exists():
            index = ConsumptionIndex(root/'training/verifications.json')
            inputs = dict(binding=binding['identity'],data=verified(root/'data_profile.json')['identity'],
                          features=verified(root/'features/train-dev.json')['identity'],seed=seed,branch=branch,
                          draw_identity=canonical_digest(schedule.tolist()),warmup=warmup['sha256'] if warmup else None,
                          numerical_code=[index.identity(Path(training.__file__).with_name(name)) for name in
                                          ('model.py','mask_adapter.py','features.py','losses.py','training.py')])
            repair_validation(directory,canonical_digest(inputs),fc,loader,dev_objects)
        result = original(binding,fc,loader,train_objects,dev_objects,schedule,seed=seed,branch=branch,warmup=warmup)
        validate_branch_receipt(result)
        return result
    training.train_phase = guarded
    try:
        yield
    finally:
        training.train_phase = original
