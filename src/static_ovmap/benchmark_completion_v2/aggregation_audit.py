"""Actual table/result-key arithmetic and hash-bound rendered-table checks."""
from pathlib import Path
import json
import math

from .common import verified, write, digest_file


def run(binding):
    root=Path(binding['output_root'])
    store=verified(root/'result_store.json')
    tables=verified(root/'tables.json')
    receipt=verified(root/'tables/receipt.json')
    if tables['result_store']!=store['identity'] or receipt['tables']!=tables['identity']:
        raise ValueError('Current table/result-store identities differ')
    rows=tables['tables']
    if {row['id'] for row in rows}!={'T1','T2','T3',*[f'S{i}' for i in range(1,9)]} or len(rows)!=11:
        raise ValueError('The eleven required tables are incomplete or duplicated')
    provenance=[json.loads(line) for line in (root/'cell_provenance.jsonl').read_text().splitlines()]
    index={(row['table'],row['row'],row['column']):row for row in provenance}
    if len(index)!=len(provenance):
        raise ValueError('A displayed cell has duplicate provenance')
    numeric,text,blank=0,0,0
    for table in rows:
        for ri,row in enumerate(table['rows']):
            for ci,cell in enumerate(row['cells']):
                key=(table['id'],ri,ci)
                if cell['value'] is None:
                    blank+=1
                    if key in index:
                        raise ValueError('A missing cell was exported as a measured value')
                    continue
                source=index[key]
                if (source['raw_value']!=cell['value'] or source['result_store']!=store['identity']
                        or source['result_keys']!=cell['results'] or not cell['results']
                        or source.get('annotation', '') != cell.get('annotation', '')):
                    raise ValueError('Actual cell provenance differs from table data')
                if any(k not in store['results'] for k in cell['results']):
                    raise ValueError('A displayed cell points to an absent source result')
                if cell.get('annotation') == '†' and any(
                        store['results'][k].get('reporting') != 'AUTHOR_REPORTED' for k in cell['results']):
                    raise ValueError('An author-reported marker points to a local or unknown result')
                if cell['unit']=='TEXT':
                    text+=1
                    continue
                values=[store['results'][k]['metrics'][cell['metric']] for k in cell['results']]
                if any(v is None for v in values) or not math.isclose(cell['value'],sum(values)/len(values),rel_tol=0,abs_tol=1e-12):
                    raise ValueError('Actual cell is not the declared independent-result mean')
                numeric+=1
    main={row['id']:row for row in rows if row['id'].startswith('T')}
    ours=main['T1']['rows'][-1]
    full=main['T2']['rows'][-1]
    # T1 AP/AP50 and T2 U11 AP/AP50 must resolve to exactly the same seeds.
    for t1,t2 in ((0,2),(1,3),(3,4),(4,5)):
        a,b=ours['cells'][t1],full['cells'][t2]
        if a['value']!=b['value'] or a['results']!=b['results']:
            raise ValueError('T1 Ours and T2 Full system use different predictions or seed means')
    visual_path=root/'review/visual_QA.json'
    visual=verified(visual_path) if visual_path.is_file() else None
    valid=bool(visual and visual['primary_inspection'] and
               {item['table'] for item in visual['artifacts']}=={'T1','T2','T3'})
    if valid:
        for item in visual['artifacts']:
            for extension,key in (('pdf','pdf_sha256'),('png','png_sha256')):
                valid &= digest_file(root/'tables'/(item['table']+'_preview.'+extension))==item[key]
            valid &= digest_file(root/'tables'/(item['table']+'.tex'))==item['tex_sha256']
    return write(root/'review/G4_actual_aggregation.json',dict(status='DONE' if valid else 'BLOCKED_DEPENDENCY',
        synthetic=False,result_store=store['identity'],tables=tables['identity'],
        table_count=len(rows),numeric_cells_checked=numeric,text_cells_source_bound=text,
        null_cells_retained=blank,actual_T1_T2_U11_keys_equal=True,
        declared_per_seed_mean_recomputed=True,rendered_three_table_primary_inspection_current=valid,
        visual_QA=None if not visual else visual['identity'],
        missing=[] if valid else ['primary visual inspection bound to current table/PDF/PNG bytes']))
