"""One locked A40 worker in the recorded frozen FC environment."""
import argparse
from pathlib import Path

from .common import verified


def main():
    import torch
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--action',required=True, choices=('features','engineer','train','nominate','repeat','holdout','predict','profile'))
    args = parser.parse_args(); binding = verified(Path(args.output_root)/'source_binding.json')
    if args.action == 'features':
        from .features import capture_dataset
        result = capture_dataset(binding)
    elif args.action == 'engineer':
        from .engineering import engineer
        result = engineer(binding)
    elif args.action in ('train','repeat','nominate'):
        from .training import train_seed, nominate
        from .resume import validation_resume_guard, verify_completed_seed
        if args.action == 'train':
            with validation_resume_guard(): result = train_seed(binding,17)
        elif args.action == 'repeat':
            with validation_resume_guard(): train_seed(binding,29)
            verify_completed_seed(binding,29); result = nominate(binding,29)
        else:
            verify_completed_seed(binding,17); result = nominate(binding,17)
    elif args.action == 'holdout':
        from .features import capture_dataset
        from .recognition import dev_recognition,holdout_recognition
        dev_recognition(binding)
        capture_dataset(binding,holdout=True); result = holdout_recognition(binding)
    elif args.action == 'predict':
        from .prediction import regression_readout
        result = regression_readout(binding)
        from .runtime_profile import cache_only_profile
        cache_only_profile(binding)
    elif args.action == 'profile':
        from .runtime_profile import cache_only_profile
        result = cache_only_profile(binding)
    print('WORKER_COMPLETE',args.action,result['identity'],flush=True)


if __name__ == '__main__': main()
