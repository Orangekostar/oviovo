"""Fixed two-seed optimization, identity-bound resume and DEV-only nomination."""
import json
import hashlib
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .common import ConsumptionIndex, arrays_record, canonical_digest, verified, write
from .model import ReadoutHead, unit
from .losses import auxiliary_losses, base_loss

BRANCHES = {'LR05_MA_8': ('NONE', False), 'LR06_MV_VIEW': ('VIEW', False),
            'LR07_MV_SURFACE': ('SURFACE', False), 'LR08_MV_AUX': ('SURFACE', True)}


def draw_schedule(objects, seed, *, updates=2000):
    classes = {}
    for index, obj in enumerate(objects):
        if obj['base']:
            classes.setdefault(int(obj['class_id']), []).append(index)
    if not classes:
        raise ValueError('Uniform base-class draw requires usable original objects')
    ids = sorted(classes); rng = np.random.default_rng(seed)
    draws = np.empty((updates*16, 3), np.int32)
    for index in range(len(draws)):
        category = ids[int(rng.integers(len(ids)))]
        choices = classes[category]
        draws[index] = choices[int(rng.integers(len(choices)))], (2,4,8)[index % 3], 1+(index//3) % 3
    return draws


def optimizer_for(head):
    decay, plain = [], []
    for name, param in head.named_parameters():
        if param.requires_grad:
            (plain if param.ndim < 2 or name == 'queries' or name.endswith('.bias') else decay).append(param)
    return torch.optim.AdamW([dict(params=decay, weight_decay=.01), dict(params=plain, weight_decay=0)],
                            lr=1e-4, betas=(.9,.999), eps=1e-8)


def learning_rate(completed, total=2000):
    if completed < 100:
        return 1e-4 * completed / 99
    progress = min(1., (completed-100) / max(1, total-101))
    return 1e-4 * (.1 + .9*.5*(1+math.cos(math.pi*progress)))


def save_checkpoint(path, head, optimizer, *, seed, branch, step, input_identity, **metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model=head.state_dict(), optimizer=optimizer.state_dict(), seed=seed, branch=branch,
                 step=step, sampler_index=step*16, input_identity=input_identity,
                 scheduler=dict(completed_updates=step, total_updates=2000, next_lr=learning_rate(step)),
                 RNG=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), **metadata)
    temp = path.with_name(path.name+'.tmp')
    torch.save(state, temp); temp.replace(path)


def restore_checkpoint(path, head, optimizer, *, input_identity):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['input_identity'] != input_identity or state['sampler_index'] != state['step']*16:
        raise ValueError('Checkpoint input identity or sampler position changed')
    head.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
    rng = state['RNG']
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
    if rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])
    return state


def model_digest(head):
    digest = hashlib.sha256()
    for name, value in head.state_dict().items():
        digest.update(name.encode()); digest.update(memoryview(value.detach().cpu().contiguous().numpy()).cast('B'))
    return digest.hexdigest()


def initialized_branch(seed, branch, warmup):
    grouping, _ = BRANCHES[branch]
    torch.manual_seed(seed)
    head = ReadoutHead(grouping)
    head.ma.load_state_dict({k.removeprefix('ma.'): v for k,v in warmup.items()}, strict=True)
    if grouping != 'NONE':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1009)
            def reset(module):
                if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                    module.reset_parameters()
            for name, module in head.named_children():
                if name != 'ma': module.apply(reset)
            torch.nn.init.normal_(head.queries, mean=0, std=.02)
    return head


def evaluate_dev(head, loader, objects, fc):
    if head is not None: head.eval()
    rows, families = [], {}
    with torch.no_grad():
        for obj in objects:
            if not obj['base']: continue
            losses, predictions = [], []
            target = fc.base_targets[obj['class_id']]
            for condition in range(4):
                inputs, vectors, _ = loader.load(obj, condition, 8)
                embedding = (head(inputs, fc.phi)['embedding'] if head is not None else
                             unit(vectors.mean(0)))
                logits = embedding @ fc.base_text.T / .07
                ce = torch.nn.functional.cross_entropy(logits[None], torch.tensor([target], device=fc.device))
                losses.append(float(ce)); predictions.append(fc.ids[int((embedding @ fc.text_tensor.T).argmax())])
            value = .5*losses[0] + sum(losses[1:])/6
            families.setdefault(obj['family'], []).append(value)
            rows.append(dict(key=obj['key'], family=obj['family'], class_id=obj['class_id'],
                             CE=losses, criterion=value, full200_top1=predictions))
    means = {k:float(np.mean(v)) for k,v in families.items()}
    criterion = float(np.mean(list(means.values())))
    if not math.isfinite(criterion): raise ValueError('Nonfinite DEV criterion')
    if head is not None: head.train()
    return dict(criterion=criterion, per_family=means, original_objects=len(rows), records=rows,
                classification_denominator='160_BASE_CLASSES', view_prefix=8, equal_family=True)


def select_earliest(rows):
    winner = None
    for row in sorted(rows, key=lambda r:r['step']):
        if winner is None or row['criterion'] < winner['criterion']-1e-12:
            winner = row
    return winner


def train_phase(binding, fc, loader, train_objects, dev_objects, schedule, *, seed, branch, warmup=None):
    root = Path(binding['output_root']); directory = root / 'training' / f'seed{seed}' / branch
    receipt_path = directory / 'receipt.json'; directory.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(root/'training/verifications.json')
    inputs = dict(binding=binding['identity'], data=verified(root/'data_profile.json')['identity'],
                  features=verified(root/'features/train-dev.json')['identity'], seed=seed, branch=branch,
                  draw_identity=canonical_digest(schedule.tolist()), warmup=warmup['sha256'] if warmup else None,
                  numerical_code=[index.identity(Path(__file__).with_name(name)) for name in
                                  ('model.py','mask_adapter.py','features.py','losses.py','training.py')])
    identity = canonical_digest(inputs)
    if receipt_path.exists():
        previous = verified(receipt_path)
        if previous['input_identity'] != identity: raise ValueError('Scientific training identity changed')
        if previous['status'] == 'COMPLETE':
            for key in ('last_checkpoint','selected_checkpoint'):
                if previous.get(key): index.identity(previous[key]['path'], previous[key])
            return previous
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if branch == 'WARMUP':
        head = ReadoutHead('NONE'); use_aux = False
    else:
        warmup_state = torch.load(warmup['path'], map_location='cpu', weights_only=False)['model']
        head = initialized_branch(seed, branch, warmup_state); use_aux = BRANCHES[branch][1]
    initial_digest = model_digest(head)
    head.to(fc.device).train()
    # Extra-layer initialization must not change the common phase RNG reset.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    optimizer = optimizer_for(head)
    last_path = directory/'last.pt'; selected_path = directory/'selected.pt'
    validations, start_step, previous_seconds = [], 0, 0.
    totals = dict(positive_tokens=0, negative_tokens=0, unknown_tokens=0, correspondence_pairs=0)
    if last_path.exists():
        state = restore_checkpoint(last_path, head, optimizer, input_identity=identity)
        start_step = state['step']; validations = state.get('validations', [])
        previous_seconds = state.get('elapsed_seconds', 0.)
        totals.update(state.get('auxiliary_totals', {}))
    elif receipt_path.exists():
        raise ValueError('Interrupted scientific phase has no valid resumable checkpoint')
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    ledger = directory/'updates.jsonl'
    interrupted_updates = []
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if row['step'] > start_step: interrupted_updates.append(row)
        if interrupted_updates:
            write(directory/f'pre_resume_{time.time_ns()}.json', dict(discarded_uncheckpointed_updates=interrupted_updates))
            ledger.write_text(''.join(json.dumps(json.loads(line))+'\n' for line in ledger.read_text().splitlines()
                                      if json.loads(line)['step'] <= start_step))
    print('TRAIN_START', seed, branch, 'resume_step', start_step, 'initial', initial_digest, flush=True)
    completed = start_step
    try:
        with ledger.open('a') as stream:
            for step in range(start_step, 2000):
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(step)
                for group in optimizer.param_groups: group['lr'] = lr
                loss_sum = 0.; aux_counts = {k:0 for k in totals}
                for object_index, prefix, condition in schedule[step*16:(step+1)*16]:
                    obj = train_objects[int(object_index)]
                    clean_inputs = loader.load(obj, 0, int(prefix))[0]
                    corrupt_inputs = loader.load(obj, int(condition), int(prefix))[0]
                    clean = head(clean_inputs, fc.phi, completed_steps=step, training_forward=True)
                    corrupt = head(corrupt_inputs, fc.phi, completed_steps=step, training_forward=True)
                    loss, _ = base_loss(clean, corrupt, fc.base_targets[obj['class_id']], fc.base_text)
                    if use_aux:
                        _, clean_target = loader.targets(obj, 0, int(prefix))
                        _, corrupt_target = loader.targets(obj, int(condition), int(prefix))
                        extra, counts = auxiliary_losses([clean,corrupt], [clean_target,corrupt_target])
                        loss = loss+extra
                        for k in totals: aux_counts[k] += counts[k]
                    if not torch.isfinite(loss): raise ValueError('Nonfinite scientific training loss')
                    (loss/16).backward(); loss_sum += float(loss.detach())/16
                    del clean_inputs, corrupt_inputs, clean, corrupt, loss
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
                if not torch.isfinite(norm): raise ValueError('Nonfinite scientific gradient norm')
                optimizer.step(); completed = step+1
                for k in totals: totals[k] += aux_counts[k]
                row = dict(step=completed, loss=loss_sum, gradient_norm=float(norm), lr=lr,
                           effective_original_objects=16, sampler_index=completed*16, auxiliary=aux_counts)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                if branch != 'WARMUP' and completed in (500,1000,1500,2000):
                    validation = evaluate_dev(head, loader, dev_objects, fc)
                    validation.update(step=completed, seed=seed, branch=branch)
                    write(directory/f'dev_{completed:04d}.json', validation)
                    validations.append(dict(step=completed, criterion=validation['criterion'],
                                            receipt=str(directory/f'dev_{completed:04d}.json')))
                    winner = select_earliest(validations)
                    if winner['step'] == completed:
                        save_checkpoint(selected_path, head, optimizer, seed=seed, branch=branch, step=completed,
                                        input_identity=identity, criterion=winner['criterion'], validations=validations)
                    print('DEV_VALIDATION',seed,branch,completed,validation['criterion'],flush=True)
                if completed % 25 == 0 or completed == 2000:
                    elapsed = previous_seconds+time.perf_counter()-started
                    save_checkpoint(last_path, head, optimizer, seed=seed, branch=branch, step=completed,
                                    input_identity=identity, validations=validations, elapsed_seconds=elapsed,
                                    auxiliary_totals=totals)
                    write(receipt_path, dict(status='RUNNING', input_identity=identity, seed=seed, branch=branch,
                          completed_steps=completed, elapsed_seconds=elapsed, validations=validations))
                    print('TRAIN_PROGRESS',seed,branch,completed,round(loss_sum,6),'seconds',round(elapsed,1),flush=True)
    except BaseException:
        if completed:
            save_checkpoint(last_path, head, optimizer, seed=seed, branch=branch, step=completed,
                            input_identity=identity, validations=validations,
                            elapsed_seconds=previous_seconds+time.perf_counter()-started, auxiliary_totals=totals)
        raise
    torch.cuda.synchronize()
    result = write(receipt_path, dict(status='COMPLETE', input_identity=identity, seed=seed, branch=branch,
                 completed_steps=2000, original_item_draws=32000, initial_model_digest=initial_digest,
                 final_model_digest=model_digest(head), trainable_parameters=sum(p.numel() for p in head.parameters()),
                 last_checkpoint=index.identity(last_path), selected_checkpoint=index.identity(selected_path) if branch != 'WARMUP' else None,
                 selected_step=select_earliest(validations)['step'] if validations else 2000,
                 DEV_criterion=select_earliest(validations)['criterion'] if validations else None,
                 validations=validations, auxiliary_totals=totals, frozen_gradient_parameters=0,
                 elapsed_seconds=previous_seconds+time.perf_counter()-started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 discarded_uncheckpointed_updates=len(interrupted_updates), input_contract=inputs))
    index.write_memo(root/'training/verifications.json')
    return result


def train_seed(binding, seed):
    from static_ovmap.backbone_wave1.runtime import exclusive_lock
    from .features import FrozenFC, ObjectLoader, execution_config
    root = Path(binding['output_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != binding['gpu']:
        raise ValueError('Training worker requires bound single GPU')
    verified(root/'engineering/receipt.json'); verified(root/'data_profile.json')
    features = verified(root/'features/train-dev.json'); split = verified(root/'split_manifest.json')
    objects = {role:[o for scene in split['roles'][role] for o in verified(root/'data/generated'/scene/'manifest.json')['objects']]
               for role in ('train','dev')}
    if seed == 17:
        branches = list(BRANCHES)
    elif seed == 29:
        nominated = verified(root/'dev_nomination.json')
        branches = ['LR05_MA_8', nominated['proposed_architecture']]
    else: raise ValueError('Only fixed scientific seeds17/29 supported')
    schedule = draw_schedule(objects['train'], seed)
    index = ConsumptionIndex(root/'training/verifications.json')
    arrays_record(root/'training'/f'seed{seed}'/'draws.npz', dict(draws=schedule), index)
    write(root/'training'/f'seed{seed}'/'draw_contract.json', dict(seed=seed,
          sampling='UNIFORM_BASE_CLASS_THEN_UNIFORM_ORIGINAL_OBJECT', object_keys=[o['key'] for o in objects['train']],
          view_cycle=[2,4,8], corruption_cycle='truncate/append/both, changing after every complete view-count cycle',
          fixed_nine_condition_prefix_pairs=True, shared_across_warmup_and_branches=True, updates_per_phase=2000))
    with exclusive_lock(execution_config(binding)['gpu_lock']):
        fc = FrozenFC(binding); loader = ObjectLoader(fc, features)
        from .engineering import frozen_digest
        frozen_before = frozen_digest(fc)
        reference_path = root/'recognition/dev_FC8.json'
        reference_key = canonical_digest(dict(data=verified(root/'data_profile.json')['identity'],features=features['identity']))
        if reference_path.exists():
            if verified(reference_path)['input_identity'] != reference_key: raise ValueError('DEV FC8 reference changed')
        else:
            write(reference_path,dict(evaluate_dev(None,loader,objects['dev'],fc),status='COMPLETE',input_identity=reference_key))
        warmup = train_phase(binding, fc, loader, objects['train'], objects['dev'], schedule, seed=seed, branch='WARMUP')
        starts = {}
        for branch in branches:
            state = torch.load(warmup['last_checkpoint']['path'], map_location='cpu', weights_only=False)['model']
            starts[branch] = model_digest(initialized_branch(seed, branch, state))
        proposed = [v for k,v in starts.items() if k != 'LR05_MA_8']
        if len(set(proposed)) > 1: raise ValueError('Matched proposed starts are not bitwise equal')
        write(root/'training'/f'seed{seed}'/'branch_initialization.json', dict(seed=seed, MA_start=warmup['last_checkpoint'],
              proposed_initialization_seed=seed+1009, model_state_digests=starts, matched_proposed_starts=True))
        rows = [train_phase(binding, fc, loader, objects['train'], objects['dev'], schedule,
                            seed=seed, branch=branch, warmup=warmup['last_checkpoint']) for branch in branches]
        frozen_after = frozen_digest(fc)
        if frozen_before != frozen_after or any(p.grad is not None for p in fc.model.parameters()):
            raise ValueError('Frozen backbone/text changed during scientific training')
        result = write(root/'training'/f'seed{seed}'/'receipt.json', dict(status='COMPLETE', seed=seed,
              warmup=warmup['identity'], branches={r['branch']:r['identity'] for r in rows},
              scientific_updates=2000*(1+len(rows)), frozen_before=frozen_before, frozen_after=frozen_after,
              architecture_selection='DEV_ONLY', physical_image_encodings=0))
        fc.index.write_memo(root/'features/verifications.json')
    return result


def nominate(binding, seed=17):
    root = Path(binding['output_root']); trained = verified(root/'training'/f'seed{seed}'/'receipt.json')
    branch_rows = [verified(root/'training'/f'seed{seed}'/name/'receipt.json') for name in trained['branches']]
    ranking = sorted(branch_rows, key=lambda r:(r['DEV_criterion'], list(BRANCHES).index(r['branch'])))
    if seed == 17:
        candidates = [r for r in branch_rows if r['branch'] != 'LR05_MA_8']
        selected = candidates[0]
        for row in sorted(candidates, key=lambda r:list(BRANCHES).index(r['branch'])):
            if row['DEV_criterion'] < selected['DEV_criterion']-1e-12:
                selected = row
        path = root/'dev_nomination.json'; architecture = selected['branch']
    else:
        path = root/'repeat_nomination.json'; architecture = verified(root/'dev_nomination.json')['proposed_architecture']
    proposed = next(r for r in branch_rows if r['branch']==architecture)
    control = next(r for r in branch_rows if r['branch']=='LR05_MA_8')
    reference = verified(root/'recognition/dev_FC8.json')
    return write(path, dict(status='NOMINATED', seed=seed, proposed_architecture=architecture,
          ranking=[dict(branch=r['branch'], DEV_criterion=r['DEV_criterion'], selected_step=r['selected_step'],
                        checkpoint=r['selected_checkpoint']) for r in ranking],
          checkpoints={r['branch']:r['selected_checkpoint'] for r in branch_rows},
          data=verified(root/'data_profile.json')['identity'], classes=verified(root/'class_split.json')['identity'],
          descriptive_DEV_criterion_margin_vs_FC8=reference['criterion']-proposed['DEV_criterion'],
          descriptive_DEV_criterion_margin_vs_MA=control['DEV_criterion']-proposed['DEV_criterion'],
          FC8_reference_identity=reference['identity'],
          holdout_annotation_reads=0, selection_uses_benchmark_AP=False, selection_uses_runtime=False))
