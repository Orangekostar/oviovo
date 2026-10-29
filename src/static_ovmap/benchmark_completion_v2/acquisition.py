"""Authorized finite asset downloads with identity receipts and disk reservation."""
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

from .common import read, verified, write, canonical_digest, digest_file, StageFailure


def download(url, destination, *, expected_sha256=None, source, authorization):
    """One initial attempt and one retry per same URL/destination identity."""
    if not authorization:
        raise StageFailure('Asset acquisition has no recorded authorization', missing=[url])
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + '.download.json')
    identity = canonical_digest(dict(url=url, destination=str(path), sha256=expected_sha256, source=source))
    prior = verified(receipt) if receipt.exists() else None
    if prior and prior['input_identity'] != identity:
        raise ValueError('Acquisition identity changed; preserve its previous receipt')
    if prior and prior['status'] == 'DONE':
        if not path.is_file() or path.stat().st_size != prior['bytes'] or digest_file(path) != prior['sha256']:
            raise ValueError('Completed downloaded asset changed')
        return prior
    attempts = (prior or {}).get('attempts', [])
    if len(attempts) >= 2:
        raise StageFailure('Asset access already failed twice for this identity', missing=[str(receipt)])
    start = time.monotonic()
    attempt = dict(number=len(attempts) + 1, url=url)
    part = path.with_name(path.name + '.part')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'OVMap-reproducibility/2', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=45) as response:
            length = response.headers.get('Content-Length')
            size = int(length) if length else None
            content_type = response.headers.get('Content-Type', '')
            attempt.update(http_status=response.status, final_url=response.url, content_length=size,
                           content_type=content_type, etag=response.headers.get('ETag'))
            if response.status != 200:
                raise ValueError('Unexpected asset HTTP response')
            if size is None:
                raise StageFailure('Asset size not declared; inspect official archive listing before streaming', missing=[url])
            if shutil.disk_usage(path.parent).free < 20 * 2**30 + 2 * size:
                raise StageFailure('Insufficient disk reserve for next asset', status='BLOCKED_RESOURCE', missing=[url])
            # Login/HTML pages cannot be treated as successful model/data archives.
            if 'text/html' in content_type.casefold() and path.suffix.lower() not in ('.html', '.htm'):
                raise StageFailure('Official asset URL returned an HTML/login/confirmation page', missing=[url])
            with part.open('wb') as stream:
                for block in iter(lambda: response.read(2**20), b''):
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if part.stat().st_size != size:
                raise ValueError('Downloaded asset byte count differs from declared Content-Length')
        digest = digest_file(part)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError('Downloaded asset differs from official expected SHA256')
        part.replace(path)
        attempt.update(status='DONE', elapsed_seconds=time.monotonic() - start)
        return write(receipt, dict(status='DONE', input_identity=identity, url=url, source=source,
            authorization=authorization, attempts=attempts + [attempt], path=str(path), bytes=path.stat().st_size,
            sha256=digest, expected_sha256=expected_sha256,
            integrity_level='AUTHOR_SHA256_VERIFIED' if expected_sha256 else 'HTTPS_SOURCE_AND_RECORDED_BYTE_IDENTITY'))
    except Exception as exc:
        status = exc.status if isinstance(exc, StageFailure) else 'BLOCKED_DEPENDENCY' if isinstance(exc, (urllib.error.URLError, TimeoutError)) else 'FAILED_RUNTIME'
        attempt.update(status=status, error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic() - start)
        write(receipt, dict(status=status, input_identity=identity, url=url, source=source,
            authorization=authorization, attempts=attempts + [attempt], path=str(path),
            partial_bytes=part.stat().st_size if part.exists() else 0))
        raise StageFailure('Asset acquisition failed: ' + str(exc), status=status, missing=[str(receipt)]) from exc


def scannet_scene(binding, scene, *, ground_truth=False):
    """Acquire only files absent from the finite authorized input inventory."""
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    if scene not in protocol['scopes']['scannet200_val']['scenes']:
        raise ValueError('Scene is outside the official locked validation scope')
    inventory = verified(root / 'protocol/scannet_asset_inventory.json')['scenes'][scene]['files']
    refreshed = root / 'assets/refreshed_scannet_inventory.json'
    if refreshed.is_file():
        inventory = dict(inventory, **verified(refreshed)['scenes'].get(scene, {}).get('files', {}))
    inherited = binding['inherited'] or {}
    authority = inherited.get('scan_authority')
    if authority is None:
        # STRUCT's downloaded split already binds the persisted user authorization.
        manifest = binding['structure_documents'].get('split_manifest.json')
        if manifest:
            authority = verified(manifest).get('data_authority')
    if not authority:
        raise StageFailure('Persisted official ScanNet data authorization is absent')
    from .common import Resolver, ConsumedInputs
    resolver = Resolver(binding['path_map'])
    authority = resolver.rewrite(authority)
    index = ConsumedInputs(root)
    index.identity(authority['path'], authority)
    authorization = read(authority['path'])
    if 'USER_EXPLICITLY_CONFIRMED' not in repr(authorization) and 'explicitly confirmed' not in repr(authorization).casefold():
        raise StageFailure('Bound data authority does not contain user-confirmed ScanNet access')
    suffixes = ('.aggregation.json', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.labels.ply') if ground_truth else ('.sens', '_vh_clean_2.ply', '.txt')
    files = {}
    for suffix in suffixes:
        if inventory.get(suffix):
            files[suffix] = dict(inventory[suffix], origin='EXISTING_AUTHORIZED_INPUT')
            continue
        version = 'v1' if suffix == '.sens' else 'v2'
        url = f'https://kaldir.vc.cit.tum.de/scannet/{version}/scans/{scene}/{scene}{suffix}'
        path = root / 'assets/datasets/scannet/scans' / scene / (scene + suffix)
        files[suffix] = download(url, path, source='official ScanNet release; persisted user authorization',
            authorization=dict(path=authority['path'], sha256=authority['sha256']))
    index.save()
    return write(root / 'assets/datasets/scannet/receipts' / (scene + ('_GT' if ground_truth else '') + '.json'),
                 dict(status='DONE', scene=scene, GT_files=ground_truth, files=files, GT_labels_opened=False))
