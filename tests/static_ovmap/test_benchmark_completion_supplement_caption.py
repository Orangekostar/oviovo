"""Actual PDF regression: a long caption must stay inside the table's margins."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from static_ovmap.benchmark_completion_v2.tables import _emit

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('supplement_caption_renderer',
    REPO / 'scripts/evaluation/render_ovmap_benchmark_completion_supplement.py')
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


@pytest.mark.parametrize(('name', 'columns'), [('S1', 4), ('S2', 6), ('S5', 7), ('S8', 4)])
def test_long_caption_stays_inside_14mm_page_margins(tmp_path, name, columns):
    if not shutil.which('pdflatex') or not shutil.which('pdftotext'):
        pytest.skip('Actual PDF regression requires pdflatex and pdftotext')
    table = dict(id=name, title='Caption layout fixture',
        caption='Caption words stay within the same page margins as the table body. ' * 8,
        headers=['Value ' + str(i) for i in range(columns - 1)],
        rows=[dict(label='A row with a sufficiently long setting label',
                   cells=[dict(value='fixture', unit='TEXT') for _ in range(columns - 1)])])
    _emit(tmp_path, table)
    original = (tmp_path / 'tables' / (name + '.tex')).read_text()
    tex = tmp_path / 'caption.tex'
    tex.write_text(RENDERER.wrapper(table, original))
    for _ in range(2):
        compiled = subprocess.run(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                                   '-halt-on-error', tex.name], cwd=tmp_path, capture_output=True, text=True)
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    extracted = subprocess.run(['pdftotext', '-bbox', str(tex.with_suffix('.pdf')), '-'],
                               capture_output=True, text=True, check=True)
    document = ET.fromstring(extracted.stdout)
    pages = document.findall('.//{*}page')
    assert len(pages) == 1
    words = pages[0].findall('.//{*}word')
    margin_pt = 14 * 72 / 25.4
    left = min(float(word.attrib['xMin']) for word in words)
    right = max(float(word.attrib['xMax']) for word in words)
    assert left >= margin_pt - 0.5, f'{name}: caption starts at {left:.3f} pt, before margin {margin_pt:.3f} pt'
    assert right <= float(pages[0].attrib['width']) - margin_pt + 0.5
    caption_start = [float(word.attrib['xMin']) for word in words
                     if word.text == 'Table' and float(word.attrib['yMin']) > 60]
    assert len(caption_start) == 1
    assert caption_start[0] <= margin_pt + 0.5, 'Long caption must use the full actual table width'
