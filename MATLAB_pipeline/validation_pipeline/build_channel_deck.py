"""Build the comparison deck of one GT unit from its for ppt report, following the ch259 blueprint
(pt 4 emu 7 ch 259/wave_clus_mcws_compare.odp): same slides, same positions, current figures, plus an
accounting slide (accounting.csv) and a flags slide (flags.csv). Writes <report>/<channel> wc_mcws_auto.pptx
and .odp; never touches any other file.

    python build_channel_deck.py "<channel>/for ppt"            # one GT unit
    python build_channel_deck.py "<channel>/for ppt/GT c3"      # a channel with several GT units
    python build_channel_deck.py "<channel>/10_19 for ppt"      # a tagged run: arm from '<channel>/python second pass 10_19'

Needs python-pptx (scratch venv) and LibreOffice (/opt/libreoffice7.5) for the .odp.
"""
import csv, glob, os, re, subprocess, sys, tempfile
from pptx import Presentation
from pptx.util import Inches, Pt

SOFFICE = '/opt/libreoffice7.5/program/soffice'


def first(pattern):
    f = sorted(glob.glob(pattern, recursive=True), key=len)
    return f[0] if f else None


def pic(slide, path, x, y, w=None, h=None):
    if not path:
        return
    kw = {}
    if w: kw['width'] = Inches(w)
    if h: kw['height'] = Inches(h)
    slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def text(slide, s, x, y, w, h, size=14, bold=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tb.word_wrap = True
    for i, line in enumerate(s.split('\n')):
        p = tb.paragraphs[0] if i == 0 else tb.add_paragraph()
        p.text = line; p.font.size = Pt(size); p.font.bold = bold and i == 0


def table(slide, rows, x, y, w, col_w, size=9):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.2 * len(rows))).table
    for r_ in t.rows: r_.height = Inches(0.2)
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(size); p.font.bold = i == 0


def build(rep):
    rep = rep.rstrip('/')
    m = re.match(r'(.*)/(?:(\S+) )?for ppt(?: (\S+))?(?:/|$)', rep)  # <channel>/[<tag> ]for ppt[ <tag>][/GT c<g>]
    chan_dir, tag = m.group(1), m.group(2) or m.group(3) or ''; chan = os.path.basename(chan_dir)
    wn = os.path.join(rep, 'wave_clus notch')
    mm = first(rep + '/membership_ch*.mat'); ch = re.search(r'membership_ch(\d+)', mm).group(1)
    acc = list(csv.DictReader(open(os.path.join(rep, 'accounting.csv'))))
    fl = list(csv.DictReader(open(os.path.join(rep, 'flags.csv')))) if os.path.exists(os.path.join(rep, 'flags.csv')) else []
    cls = {r['comparison'] + '|' + r['region']: r for r in acc}
    gt_c = re.search(r'gt class c(\d+)', ' '.join(r['region'] for r in acc)).group(1)
    wc_c = re.search(r'wc_notch class c(\d+)', ' '.join(r['region'] for r in acc)).group(1)
    py_c = re.search(r'python class c(\d+)', ' '.join(r['region'] for r in acc)).group(1)
    arm_dir = (os.path.join(chan_dir, f'run {tag}') if tag[:1].isalpha() else os.path.join(chan_dir, f'python second pass {tag}')) if tag else chan_dir
    py_arm = os.path.basename(sorted(glob.glob(arm_dir + '/python_recluster_*_second_pass'))[-1])

    def n(cmp, reg, f='n_spikes'):
        return cls.get(cmp + '|' + reg, {}).get(f, '?')

    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = chan + (f' ({tag} run)' if tag else '')
    s.placeholders[1].text = (f'GT c{gt_c}  |  wave_clus notch c{wc_c}  |  python c{py_c} ({py_arm})\n'
                              'wave_clus and python both with GT\'s notch')
    # 2 wave_clus
    s = prs.slides.add_slide(blank)
    f2p = sorted(glob.glob(wn + '/01_metrics/fig2print*.png'))
    pic(s, next((f for f in f2p if not f.endswith('a.png')), None), 0.2, 0.3, 11.0, 7.0)
    pic(s, next((f for f in f2p if f.endswith('a.png')), None), 10.5, 1.8, 2.7, 4.8)
    text(s, f'Wave_clus (notch)\nmatched class c{wc_c}: {n("wave_clus notch vs python", f"wc_notch class c{wc_c}: regions sum")} spikes',
         10.4, 0.1, 2.9, 1.0, 14, True)
    # 3 python
    s = prs.slides.add_slide(blank)
    pic(s, first(rep + '/01_metrics/python/python_cluster_page-1.png'), 1.2, 1.5, 11.2, 5.9)
    text(s, f'Python ({py_arm}) -- matched class c{py_c}: {n("wave_clus notch vs python", f"python class c{py_c}: regions sum")} spikes',
         1.2, 0.5, 11.0, 0.8, 18, True)
    # 4 wave_clus vs python venn + class matches
    s = prs.slides.add_slide(blank)
    v = wn + '/02_venns/wave_clus_vs_python/'
    pic(s, first(v + 'venn3_*_cluster.png'), 4.8, 2.0, 4.2, 5.0)
    pic(s, first(v + 'class_match_*_python_sd1_vs_wave_clus*.png'), 8.8, 0.0, 4.5, 2.5)
    pic(s, first(v + 'class_match_*_wave_clus*_vs_python_sd1.png'), 0.0, 0.0, 4.5, 2.6)
    c = 'wave_clus notch vs python'
    text(s, (f'wave_clus notch c{wc_c} vs python c{py_c}\nwave_clus only {n(c, "ref only")} | wave_clus x quarantine '
             f'{n(c, "ref x quarantine")} | matched {n(c, "matched (ref spikes)")}\npython only {n(c, "arm only")} | '
             f'rescued only {n(c, "rescued only")} | quarantine only {n(c, "quarantine only")}'), 0.1, 3.0, 4.6, 2.0, 12)
    # 5 full rasters wave_clus | python (08)
    s = prs.slides.add_slide(blank)
    r8 = wn + '/08_rasters_wave_clus_vs_python/'
    pic(s, first(r8 + 'wave_clus notch_stimulus_rasters_*resp_window*fig 1.png'), 0.0, 0.2, 6.0, 7.5)
    pic(s, first(r8 + 'python_sd1_stimulus_rasters_*resp_window*fig 1.png'), 7.2, 0.2, 6.0, 7.5)
    # 6 per-stimulus summary (08)
    s = prs.slides.add_slide(blank)
    pic(s, first(r8 + '*_gt_vs_arm_summary.png'), 1.4, 0.0, 9.9, 7.5)
    # 7 quarantine set: wave_clus x python quarantine
    s = prs.slides.add_slide(blank)
    q = wn + '/05_wave_clus_x_quarantine/'
    pic(s, first(q + '00_*/*resp_window*fig 1.png'), 6.8, 0.2, 6.0, 7.5)
    pic(s, first(q + 'waveforms_by_quarantine_property.png'), 0.1, 1.2, 6.6)
    text(s, f'Quar set: wave_clus c{wc_c} spikes python quarantined, not in python c{py_c} ({n(c, "ref x quarantine")})',
         0.1, 0.2, 6.6, 0.9, 14, True)
    # 8 GT
    s = prs.slides.add_slide(blank)
    pic(s, first(rep + '/01_metrics/gt/*plots001.png'), 1.5, 1.3, 11.1)
    text(s, f'GT (hand selected) -- class c{gt_c}: {n("GT vs python", f"gt class c{gt_c}: regions sum")} spikes', 1.5, 0.4, 11.0, 0.8, 18, True)
    # 9 GT venns + class matches
    s = prs.slides.add_slide(blank)
    gp = rep + '/02_venns/gt_vs_python/'; gw = wn + '/02_venns/gt_vs_wave_clus/'
    pic(s, first(gp + 'venn3_*_cluster.png'), 6.2, 1.5, 4.2, 5.0)
    pic(s, first(gw + 'venn3_*_cluster.png'), 2.5, 1.7, 4.0, 4.8)
    pic(s, first(gw + 'class_match_*_wave_clus_vs_GT.png'), 0.0, 0.0, 3.5, 2.0)
    pic(s, first(gw + 'class_match_*_GT_vs_wave_clus.png'), 0.0, 5.2, 3.9, 2.2)
    pic(s, first(gp + 'class_match_*_GT_vs_python_sd1.png'), 9.2, 5.2, 4.1, 2.2)
    pic(s, first(gp + 'class_match_*_python_sd1_vs_GT.png'), 9.5, 0.0, 3.8, 1.9)
    # 10 python only / wave_clus only vs GT
    s = prs.slides.add_slide(blank)
    pic(s, first(rep + '/09_gt_vs_python/02_python_only/python_only/*resp_window*fig 1.png'), 0.0, 0.0, 6.0, 7.5)
    pic(s, first(wn + '/09_gt_vs_wave_clus/02_wave_clus_only/wave_clus_only/*resp_window*fig 1.png'), 7.2, 0.0, 6.0, 7.5)
    text(s, f'python only vs GT ({n("GT vs python", "arm only")})', 0.1, 7.05, 6.0, 0.4, 11, True)
    text(s, f'wave_clus only vs GT ({n("GT vs wave_clus notch", "arm only")})', 7.3, 7.05, 6.0, 0.4, 11, True)
    # 11 accounting
    s = prs.slides.add_slide(blank)
    text(s, 'Bookkeeping: venn regions = subsets; regions add up to each class; in-window counts (each sort in its own '
            'full-run windows) add up to its raster page', 0.3, 0.1, 12.7, 0.6, 13, True)
    rows = [['comparison', 'region', 'spikes', 'venn', 'in window', 'check']]
    rows += [[r['comparison'], r['region'], r['n_spikes'], r['expected'], r['n_in_window'], r['check']] for r in acc]
    table(s, rows, 0.3, 0.7, 12.7, [3.0, 3.6, 1.4, 1.4, 1.6, 1.0], 7)
    # 12 flags
    s = prs.slides.add_slide(blank)
    text(s, 'Flags -- what needs a look, and the likely reason', 0.3, 0.1, 12.7, 0.5, 16, True)
    if fl:
        rows = [['', 'comparison', 'finding', 'likely explanation', 'figures']]
        rows += [[f['severity'], f['comparison'], f['finding'], f['likely_explanation'], f['figures']] for f in fl]
        table(s, rows, 0.2, 0.7, 12.9, [0.6, 1.8, 3.9, 4.9, 1.7], 8)
    else:
        text(s, 'no flags', 0.3, 0.8, 12, 0.5, 12)

    out = os.path.join(rep, f'{chan}{" " + tag if tag else ""} wc_mcws_auto.pptx')
    prs.save(out)
    prof = tempfile.mkdtemp(prefix='lo_')
    subprocess.run([SOFFICE, f'-env:UserInstallation=file://{prof}', '--headless', '--convert-to', 'odp',
                    '--outdir', rep, out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('[deck]', out, '+ .odp' if os.path.exists(out[:-5] + '.odp') else '(no .odp)')


if __name__ == '__main__':
    for r in sys.argv[1:]:
        build(r)
