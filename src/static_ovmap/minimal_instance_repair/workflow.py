"""Ordered scientific phases; resume executes missing work instead of status output."""

from concurrent.futures import ProcessPoolExecutor


def _observe(job):
    from .reconcile import observe
    return observe(*job)


def run_phase(binding, phase, *, resume):
    if phase == 'observe':
        workers = binding['specification']['resources']['CPU_observation_workers_max']
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_observe,[(binding,s) for names in binding['cohorts'].values() for s in names]))
    if phase == 'propose':
        from .reconcile import propose
        return [propose(binding,s) for names in binding['cohorts'].values() for s in names]
    if phase == 'diagnose':
        from .diagnostics import diagnose_study
        return diagnose_study(binding)
    if phase in ('assets','pilot','recognize'):
        from .recognition import run_recognition_phase
        return run_recognition_phase(binding,phase,resume=resume)
    if phase == 'freeze':
        from .recognition import freeze
        return freeze(binding)
    if phase == 'predict':
        from .outputs import predict_study
        return predict_study(binding)
    if phase == 'evaluate':
        from .evaluation import evaluate_study
        return evaluate_study(binding)
    if phase == 'select':
        from .selection import select_study
        return select_study(binding)
    if phase == 'time':
        from .timing import time_study
        return time_study(binding)
    if phase in ('tables','publish'):
        from .reporting import run_reporting_phase
        return run_reporting_phase(binding,phase)
    raise ValueError('unknown prescribed phase: '+phase)
