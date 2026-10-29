"""Snapshot only the inherited numerical inputs consumed by frozen readouts."""
import math
from pathlib import Path

from .common import Resolver, verified, write, digest_file, source_identity


def prepare(binding):
    root = Path(binding['output_root'])
    standard = [path for path in sorted((root/'benchmark/predictions').glob('*/*/receipt.json'))
                if verified(path).get('GLOBAL')]
    legacy_path = root/'legacy/prediction_lock.json'
    if not standard and not legacy_path.exists():
        return None
    resolver = Resolver(binding.get('path_map'))
    inherited = binding['inherited']

    def checked(descriptor, *, sealed):
        path = Path(resolver.resolve(descriptor['path']))
        sha = digest_file(path)
        size = path.stat().st_size
        if ((descriptor.get('sha256') and descriptor['sha256'] != sha)
                or (descriptor.get('bytes') is not None and descriptor['bytes'] != size)):
            raise ValueError('Replay input differs from declared file: ' + str(path))
        value = verified(path) if sealed else None
        if sealed and descriptor.get('identity') and descriptor['identity'] != value['identity']:
            raise ValueError('Replay input differs from declared identity: ' + str(path))
        return path, dict(path=str(path), bytes=size, sha256=sha), value

    def snapshot(descriptor, *, sealed):
        path, original, value = checked(descriptor, sealed=sealed)
        source = 'publication/readout_inputs/' + original['sha256'] + path.suffix
        dest = root/source
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(dest.suffix+'.tmp')
            temp.write_bytes(path.read_bytes())
            temp.replace(dest)
        if digest_file(dest) != original['sha256']:
            raise ValueError('Immutable replay snapshot differs: ' + source)
        return dict(source=source, original=original,
                    original_identity=value['identity'] if sealed else None)

    def context(scene, *, include_scores):
        _, descriptor, value = checked(inherited['scenes'][scene]['context'], sealed=True)
        temperatures = {name: value['temperatures'][name] for name in ('N', 'Q', 'F')}
        if any(isinstance(t, bool) or not isinstance(t, (int, float))
               or not math.isfinite(t) or t <= 0 for t in temperatures.values()):
            raise ValueError('Replay needs the actual finite positive source temperatures')
        result = dict(template_scene=scene, context=dict(descriptor, identity=value['identity']),
                      temperatures=temperatures)
        if include_scores:
            result.update(text=snapshot(value['FC_text'], sealed=False),
                          sources={name: snapshot(value['sources'][name], sealed=True)
                                   for name in ('N', 'Q')})
        return result

    standards = {}
    for benchmark in sorted({path.parent.parent.name for path in standard}):
        cohort = 'replica8' if benchmark == 'replica48' else 'scannet_cf18'
        # Exactly the same first-cohort context used by benchmark.template_context.
        standards[benchmark] = context(inherited['cohorts'][cohort][0], include_scores=False)
    legacies = {}
    if legacy_path.exists():
        lock = verified(legacy_path)
        for scene, identity in sorted(lock['scenes'].items()):
            receipt = verified(root/'legacy/predictions'/scene/'receipt.json')
            if receipt['identity'] != identity:
                raise ValueError('Legacy replay scene differs from its prediction lock: ' + scene)
            legacies[scene] = dict(context(scene, include_scores=True), prediction_receipt=identity)
    return write(root/'publication/readout_replay.json', dict(status='DONE',
        scope='FROZEN_READOUT_REPLAY_INPUT_SNAPSHOT', standard=standards, legacy=legacies,
        calibration_inputs=[name for name in ('A/import.json', 'A/import.npz') if (root/name).is_file()],
        readout_cache_layout='readout_cache/<input_identity>.npz + .json',
        standard_inputs='benchmark/F/<benchmark>/<scene>/vectors.npz and benchmark/NQ/*/*/sources.json',
        producer=source_identity(('readout_replay',)),
        new_model_forwards=0, new_training_updates=0, raw_GT_RGB_mesh_weights_copied=False))
