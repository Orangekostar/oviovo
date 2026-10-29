"""Verified primary-paper values; never presented as local executions."""
from pathlib import Path
from .common import verified, write, digest_file


def build(binding):
    root = Path(binding['output_root'])
    sources = {
        'OpenMask3D': dict(url='https://arxiv.org/pdf/2306.13631', table='Tables1,2; Sec4.1.1',
            variant='Mask3D trained on ScanNet200 + SAM ViT-H + CLIP-L14@336; author full method',
            scannet200_val=[15.4, 19.9, 23.1], replica48=[13.1, 18.4, 24.2]),
        'Open3DIS': dict(url='https://openaccess.thecvf.com/content/CVPR2024/papers/Nguyen_Open3DIS_Open-Vocabulary_3D_Instance_Segmentation_with_2D_Mask_Guidance_CVPR_2024_paper.pdf',
            table='Tables2,4; Sec4.1', variant='ISBNet + Grounded-SAM, 2D and3D, refined pointwise feature',
            scannet200_val=[23.7, 29.4, 32.8], replica48=[18.5, 24.5, 28.2]),
        'MaskClustering': dict(url='https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_MaskClustering_View_Consensus_based_Mask_Graph_Clustering_for_Open-Vocabulary_3D_CVPR_2024_paper.pdf',
            table='Table2 semantic columns; Sec4.1', variant='CropFormer + complete mask graph + CLIP-H14; semantic instance task',
            scannet200_val=[12.0, 23.3, 30.1], replica48=None),
    }
    results = {}
    protocol = verified(root / 'protocol_lock.json')
    for method, source in sources.items():
        pdf = root / 'assets/papers' / (method + '.pdf')
        receipt = verified(pdf.with_name(pdf.name + '.download.json'))
        if receipt['status'] != 'DONE' or digest_file(pdf) != receipt['sha256']:
            raise ValueError('Downloaded primary-paper evidence changed')
        for benchmark in ('scannet200_val', 'replica48'):
            values = source[benchmark]
            if values is None:
                continue
            scope = protocol['scopes'][benchmark]
            replica_2d_only = method == 'Open3DIS' and benchmark == 'replica48'
            results[method + ':' + benchmark] = dict(status='DONE', method=method, benchmark=benchmark,
                metrics=dict(zip(('apall', 'ap50', 'ap25'), [v / 100 for v in values], strict=True)),
                reporting='AUTHOR_REPORTED', locally_executed=False, metric_unit='FRACTION',
                source=dict(url=source['url'], table='Table4; Sec4.2 Setting2 (Replica)'
                    if replica_2d_only else source['table'], pdf_sha256=receipt['sha256']),
                variant='Grounded-SAM + 2D-guided proposals + refined pointwise features; no class-agnostic 3D proposal network (2D-only)'
                    if replica_2d_only else source['variant'], protocol=protocol['identity'], scenes=scope['scenes'],
                valid_instance_ids=scope['instance_ids'], point_domain=scope['target_point_order'],
                confidence_policy='UNCHANGED_AUTHOR_EVALUATION', re_scored_on_host=False,
                full_scene_coverage='AUTHOR_REPORTED_FULL_OFFICIAL_SCOPE',
                compatible_frozen_variant=not (benchmark == 'replica48' and method in ('OpenMask3D', 'Open3DIS')),
                variant_note='Replica paper value uses Mask3D200; separately planned arbitrary-scene checkpoint port must not inherit this value'
                    if method == 'OpenMask3D' and benchmark == 'replica48' else
                    'Replica Setting2 omits the 3D proposal network; its 2D-only value cannot fill the frozen ISBNet+GSAM full-method cell'
                    if replica_2d_only else None)
    return write(root / 'external/author_reported.json', dict(status='DONE', results=results,
        scope='PRIMARY_PAPER_REFERENCE;DAG_EXECUTION_COVERAGE_IS_SEPARATE',
        OVI_51_as_Replica48_forbidden=True, Open3DIS_2D_only_as_full_variant_forbidden=True))
