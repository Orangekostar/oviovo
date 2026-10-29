#!/usr/bin/env python3
"""Compile S1--S8 from frozen table bodies, with pagination and source checks."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

repository = Path(__file__).resolve().parents[2]
for source in (repository, repository / 'src'):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from static_ovmap.benchmark_completion_v2.common import digest_file, verified, write
from static_ovmap.benchmark_completion_v2.tables import _fmt, _tex


def snapshot(paths):
    return {str(path): dict(sha256=digest_file(path), mtime_ns=path.stat().st_mtime_ns)
            for path in paths}


def wrapper(table, original):
    """Keep the original caption, header, panels and data-row bytes."""
    lines = original.splitlines()
    top = lines.index(r'\toprule')
    bottom = lines.index(r'\bottomrule')
    assert lines[top + 2] == r'\midrule'
    header = lines[top + 1]
    body = lines[top + 3:bottom]
    expected_rows = [' & '.join(_tex(value) for value in
                     [row['label'], *[_fmt(cell) for cell in row['cells']]]) + r' \\'
                     for row in table['rows']]
    actual_rows = [line for line in body if line not in (r'\midrule',)
                   and not line.startswith(r'\multicolumn')]
    assert actual_rows == expected_rows, 'Original TeX and frozen table rows differ'
    caption = next(line for line in lines if line.startswith(r'\caption{'))
    label = next(line for line in lines if line.startswith(r'\label{'))
    count = len(table['headers']) + 1
    if table['id'] == 'S1':
        widths = [.14, .25, .25, .30]
    elif table['id'] == 'S8':
        widths = [.42, .12, .16, .24]
    else:
        first = .43 if max(len(row['label']) for row in table['rows']) > 30 else .26
        widths = [first] + [(0.90 - first) / (count - 1)] * (count - 1)
    assert len(widths) == count
    columns = ''.join(r'>{\ragged' + ('right' if index == 0 or table['id'] in ('S1', 'S8') else 'left')
                      + r'\arraybackslash}p{' + f'{width:.6f}' + r'\linewidth}'
                      for index, width in enumerate(widths))
    # The left-aligned table is narrower than textwidth. Its centered caption
    # must use the same p-column widths and interior column padding.
    caption_width = '+'.join(f'{width:.6f}' + r'\linewidth' for width in widths)
    caption_width += '+' + str(2 * (count - 1)) + r'\tabcolsep'
    name = table['id']
    return '\n'.join([
        r'\documentclass[10pt]{article}',
        r'\usepackage[a4paper,landscape,margin=14mm]{geometry}',
        r'\usepackage[T1]{fontenc}', r'\usepackage{lmodern,booktabs,array,longtable}',
        r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.12}',
        r'\setlength{\LTleft}{0pt}', r'\setlength{\LTright}{\fill}',
        r'\makeatletter',
        r'\renewcommand{\LT@makecaption}[3]{%',
        r'  \LT@mcol\LT@cols{@{}l@{}}{\parbox[t]\LTcapwidth{%',
        r'    \reset@font #1{#2: }#3\endgraf\vskip\baselineskip}}}',
        r'\makeatother',
        r'\renewcommand{\thetable}{' + name + '}',
        r'\begin{document}',
        r'\setlength{\LTcapwidth}{\dimexpr' + caption_width + r'\relax}',
        r'\small',
        r'\section*{Supplementary Table ' + name + ': ' + _tex(table['title']) + '}',
        r'\begin{longtable}{@{}' + columns + r'@{}}',
        caption + label + r' \\', r'\toprule', header, r'\midrule',
        r'\endfirsthead',
        r'\multicolumn{' + str(count) + r'}{l}{\textit{Supplementary Table ' + name + r' (continued)}} \\',
        r'\toprule', header, r'\midrule', r'\endhead',
        r'\midrule', r'\multicolumn{' + str(count)
        + r'}{r}{\footnotesize Continued on next page} \\', r'\endfoot',
        r'\bottomrule', r'\endlastfoot', *body,
        r'\end{longtable}', r'\end{document}', '',
    ])


def render(root):
    tables_path = root / 'tables.json'
    tables = verified(tables_path)
    store = verified(root / 'result_store.json')
    assert tables['result_store'] == store['identity']
    selected = [table for table in tables['tables'] if table['id'].startswith('S')]
    assert [table['id'] for table in selected] == [f'S{i}' for i in range(1, 9)]
    render_paths = [tables_path, root / 'result_store.json']
    render_paths += [root / 'tables' / (table['id'] + suffix)
                     for table in selected for suffix in ('.tex', '.md')]
    source_paths = render_paths + [root / 'tables/receipt.json',
                    root / 'review/G4_actual_aggregation.json', root / 'review/visual_QA.json']
    before = snapshot(source_paths)
    script_sha = digest_file(__file__)
    receipt_path = root / 'supplement/preview_receipt.json'
    if receipt_path.is_file():
        previous = verified(receipt_path)
        if previous['tables'] == tables['identity'] and previous['script_sha256'] == script_sha:
            for path in render_paths:
                assert previous['protected_source_files'][str(path)]['sha256'] == before[str(path)]['sha256'], str(path)
            for record in previous['previews']:
                for item in record['published_files'] + record['page_images']:
                    assert digest_file(item['path']) == item['sha256'], item['path']
            assert snapshot(source_paths) == before
            print('REUSED_SUPPLEMENT_PREVIEWS', previous['identity'], flush=True)
            return previous
    archive = root / 'supplement/previews' / tables['identity'] / script_sha[:16]
    if archive.exists():
        raise RuntimeError('Incomplete preview archive exists; inspect it before another launch')
    archive.mkdir(parents=True)
    commands = []

    def run(argv, cwd):
        started = time.monotonic()
        child = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        commands.append(dict(argv=argv, actual_child_exit=child.returncode,
                             elapsed_seconds=time.monotonic() - started))
        if child.returncode:
            (archive / 'failed_command.log').write_text(child.stdout + child.stderr)
            raise RuntimeError('Preview command failed: ' + argv[0])
        return child.stdout

    previews = []
    for table in selected:
        name = table['id']
        original = root / 'tables' / (name + '.tex')
        tex = archive / (name + '_preview.tex')
        tex.write_text(wrapper(table, original.read_text()))
        compile_output = ''
        for _ in range(2):
            compile_output += run(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                                   '-halt-on-error', tex.name], archive)
        (archive / (name + '_compile.log')).write_text(compile_output)
        assert not re.search(r'Overfull|^!', compile_output, flags=re.MULTILINE), name
        pdf = tex.with_suffix('.pdf')
        info = run(['pdfinfo', str(pdf)], archive)
        pages = int(re.search(r'^Pages:\s+(\d+)', info, flags=re.MULTILINE).group(1))
        extracted = run(['pdftotext', '-layout', str(pdf), '-'], archive)
        compact = ''.join(extracted.split())
        labels = Counter(''.join(row['label'].split()) for row in table['rows'])
        assert all(compact.count(label) >= count for label, count in labels.items()), name
        page_dir = archive / (name + '_pages')
        page_dir.mkdir()
        run(['pdftoppm', '-scale-to', '1600', '-png', str(pdf), str(page_dir / 'page')], archive)
        images = sorted(page_dir.glob('page-*.png'))
        assert len(images) == pages and pages > 0
        shutil.copyfile(images[0], tex.with_suffix('.png'))
        published = []
        for suffix in ('.tex', '.pdf', '.png'):
            source = tex.with_suffix(suffix)
            target = root / 'tables' / source.name
            if not target.exists() or digest_file(target) != digest_file(source):
                shutil.copyfile(source, target)
            published.append(dict(path=str(target), sha256=digest_file(target)))
        target_log = root / 'tables' / (name + '_compile.log')
        shutil.copyfile(archive / target_log.name, target_log)
        published.append(dict(path=str(target_log), sha256=digest_file(target_log)))
        previews.append(dict(table=name, source_tex_sha256=digest_file(original),
            source_rows=len(table['rows']), caption_header_panels_and_data_rows_preserved=True,
            pages=pages, overfull_boxes=False, all_source_row_labels_in_extracted_PDF=True,
            published_files=published,
            page_images=[dict(path=str(path), sha256=digest_file(path)) for path in images]))
        print('COMPILED_SUPPLEMENT', name, 'ROWS', len(table['rows']), 'PAGES', pages, flush=True)
    assert snapshot(source_paths) == before
    assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules
    record = dict(status='DONE', scope='FROZEN_SUPPLEMENTARY_TABLE_RENDER_ONLY',
        tables=tables['identity'], result_store=store['identity'], previews=previews,
        protected_source_files=before, source_rows=sum(row['source_rows'] for row in previews),
        actual_commands=commands, script_sha256=script_sha,
        original_table_emitter_sha256=digest_file(repository / 'src/static_ovmap/benchmark_completion_v2/tables.py'),
        new_model_calls=0, new_training_updates=0, scientific_values_modified=False,
        visual_QA='PENDING_PRIMARY_INSPECTION', C1=False, C2=False, full_objective_complete=False)
    receipt = write(archive / 'receipt.json', record)
    write(receipt_path, record)
    print('SUPPLEMENT_PREVIEWS', receipt['identity'], 'TABLES', len(previews), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    (root / 'supplement').mkdir(exist_ok=True)
    with (root / 'supplement/preview.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        render(root)


if __name__ == '__main__':
    main()
