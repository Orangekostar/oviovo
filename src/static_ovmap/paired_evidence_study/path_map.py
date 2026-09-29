"""Explicit recoverable aliases for relocated immutable evidence roots."""

import json
from pathlib import Path


def apply_path_map(path):
    """Preserve embedded original paths without rewriting signed JSON payloads.

    Only explicitly supplied, absent root names may be created as symlinks.
    Existing directories/files are never replaced. Subsequent InputIndex reads
    still compare referenced bytes/SHA256 against the original receipts.
    """
    mapping = json.loads(Path(path).read_text())
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('path-map requires an old-root/new-root JSON object')
    records = []
    forbidden = {Path('/'), Path('/home'), Path('/home/ww'), Path('/mnt'), Path('/mnt/shared'), Path('/mnt/shared/ww')}
    for old, new in mapping.items():
        original, target = Path(old), Path(new).resolve(strict=True)
        if not original.is_absolute() or '..' in original.parts or original in forbidden or not target.is_dir():
            raise ValueError('path-map must name specific absolute evidence directories')
        if original.exists() or original.is_symlink():
            if original.resolve() != target:
                raise ValueError(f'path-map would replace existing content: {original}')
            created = False
        else:
            if not original.parent.is_dir():
                raise ValueError(f'original parent must exist; no broad directory creation: {original.parent}')
            original.symlink_to(target, target_is_directory=True)
            created = True
        records.append({'old_root': str(original), 'new_root': str(target), 'alias_created': created,
                        'verification': 'all consumed bound inputs still verified by original content hashes'})
    return records
