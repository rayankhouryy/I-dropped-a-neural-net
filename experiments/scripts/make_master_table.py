"""
Unified architecture-aware factorization table.

Pulls measurements from all the architecture-pairing JSONs we've produced
and emits one master table that goes into the paper:

  results/architecture_aware_master_table.{json,tex}

Rows: every (model, stage_or_path, n, branch_product, trained metrics,
random-init metrics).
"""
import json
from pathlib import Path

Path("results").mkdir(exist_ok=True)


def load_json(p, default=None):
    if not Path(p).exists():
        return default
    return json.load(open(p))


# ------------------- gather measurements -------------------
rows = []

# 1. GPT-2 family MLP
g_mlp = load_json('results/gpt2_mlp_pairing.json', {'pretrained': []})
for r in g_mlp.get('pretrained', []):
    rows.append({
        'family': 'GPT-2',
        'model':  r['model'],
        'arch':   'Transformer MLP',
        'product': r'W_2 W_1',
        'n': r['n_layers'],
        'pair_acc': r['diag_dominance']['pair_acc'],
        'auc': None,  # not stored in this older JSON
        'sep': r['diag_dominance']['pair_sep'],
        'neg_trace': r['trace_analysis']['frac_negative'],
        'random_pair_acc': None,
    })
# Append the random init baseline for gpt2 small
g_mlp_rand = g_mlp.get('random_init_baseline')
if g_mlp_rand is not None:
    rows.append({
        'family': 'GPT-2', 'model': 'gpt2 (random init)',
        'arch': 'Transformer MLP', 'product': r'W_2 W_1',
        'n': 12, 'pair_acc': g_mlp_rand['pair_acc'],
        'auc': None, 'sep': None,
        'neg_trace': g_mlp_rand['frac_negative'],
        'random_pair_acc': g_mlp_rand['pair_acc'],
    })

# 2. GPT-2 family attention
g_attn = load_json('results/gpt2_attention_pairing.json', {'pretrained': []})
for r in g_attn.get('pretrained', []):
    for path_key, path_label, prod in [
        ('VO_full',  'Attention V<->O',  r'W_O W_V'),
        ('QK',       'Attention Q<->K',  r'W_Q W_K^T'),
    ]:
        d = r[path_key]
        tr = (r['VO_trace'] if path_key == 'VO_full' else r['QK_trace'])
        rows.append({
            'family': 'GPT-2',
            'model':  r['model'],
            'arch':   path_label,
            'product': prod,
            'n':       r['n_layers'],
            'pair_acc': d['pair_acc'],
            'auc':      None,
            'sep':      d['pair_sep'],
            'neg_trace': tr['frac_negative'],
            'random_pair_acc': None,
        })

# 3. ResNet family (BasicBlock + Bottleneck) -- from factorization ablation
res_abl = load_json('results/resnet_factorization_ablation.json', [])
res_rand = load_json('results/resnet_triple_random_baseline.json', [])
rand_idx = {(r['model'], r['stage'], r['mode']): r for r in res_rand}
for r in res_abl:
    if r['mode'] != 'channel_sum':
        continue
    rb = rand_idx.get((r['model'], r['stage'], r['mode']))
    rows.append({
        'family': 'ResNet (ImageNet)',
        'model':  f"{r['model']}/{r['stage']}",
        'arch':   'Bottleneck' if r['block_type'] == 'bottleneck' else 'BasicBlock',
        'product': r'W_3 W_2 W_1' if r['block_type'] == 'bottleneck' else r'W_2 W_1',
        'n':       r['n'],
        'pair_acc': r['correct_triple']['pair_acc'],
        'auc':      r['correct_triple']['auc'],
        'sep':      r['correct_triple']['sep'],
        'neg_trace': None,
        'random_pair_acc': (rb['triple_mean_acc'] if rb else None),
    })

# 4. Modern vision: ConvNeXt + ViT
mv = load_json('results/modern_vision_pairing.json', {})
cnx = mv.get('convnext_tiny')
if cnx:
    # the stage3 result is the headline; include all stages
    for sname, sdata in cnx['stages'].items():
        rows.append({
            'family': 'Modern Vision',
            'model':  f"convnext_tiny/{sname}",
            'arch':   'ConvNeXt CNBlock',
            'product': r'W_2 W_1',
            'n':       sdata['mlp']['n'],
            'pair_acc': sdata['mlp']['pair_acc'],
            'auc':      sdata['mlp']['auc'],
            'sep':      sdata['mlp']['pair_sep'],
            'neg_trace': sdata['trace']['frac_negative'],
            'random_pair_acc': None,
        })

# Refined ConvNeXt random baseline (20 seeds)
cnx_rand = load_json('results/convnext_random_extended.json')
if cnx_rand:
    rows.append({
        'family': 'Modern Vision',
        'model':  'convnext_tiny/stage3 (random init x20)',
        'arch':   'ConvNeXt CNBlock',
        'product': r'W_2 W_1',
        'n':       cnx_rand['n_blocks'],
        'pair_acc': cnx_rand['random_aggregate']['pair_acc']['mean'],
        'auc':      cnx_rand['random_aggregate']['auc']['mean'],
        'sep':      cnx_rand['random_aggregate']['pair_sep']['mean'],
        'neg_trace': None,
        'random_pair_acc': cnx_rand['random_aggregate']['pair_acc']['mean'],
        'is_random_baseline': True,
    })

vit = mv.get('vit_b_16')
if vit:
    for k, label, prod in [
        ('mlp', 'ViT MLP',           r'W_2 W_1'),
        ('vo',  'ViT Attn V<->O',    r'W_O W_V'),
        ('qk',  'ViT Attn Q<->K',    r'W_Q W_K^T'),
    ]:
        d = vit[k]
        rows.append({
            'family': 'Modern Vision',
            'model':  'vit_b_16',
            'arch':   label,
            'product': prod,
            'n':       vit['n_layers'],
            'pair_acc': d['pair_acc'],
            'auc':      d['auc'],
            'sep':      d['pair_sep'],
            'neg_trace': d['trace']['frac_negative'],
            'random_pair_acc': mv['vit_b_16_random'][k]['mean_acc'],
        })

# 5. Synthetic ResNet sweep (already in paper; just include the headline)
# Not adding here -- it's covered in Section 5.

# ------------------- emit JSON master -------------------
with open('results/architecture_aware_master_table.json', 'w') as f:
    json.dump(rows, f, indent=2)

# ------------------- pretty print -------------------
print(f"{'Family':<20s} {'Model':<32s} {'Arch':<22s} {'n':>3s} {'pair_acc':>9s} {'AUC':>6s} {'neg_tr':>7s} {'chance':>7s}")
print('-' * 110)
for r in rows:
    chance = 1.0 / r['n'] if r['n'] else None
    nt = f"{r['neg_trace']:.0%}" if r['neg_trace'] is not None else '   --'
    auc = f"{r['auc']:.3f}" if r['auc'] is not None else '   --'
    ch  = f"{chance:.1%}" if chance is not None else ' --'
    print(f"{r['family']:<20s} {r['model']:<32s} {r['arch']:<22s} "
          f"{r['n']:>3d} {r['pair_acc']:>8.0%}  {auc:>6s} {nt:>7s} {ch:>7s}")

# ------------------- LaTeX table -------------------
# Only the headline rows: best stage per model, no random baselines.
HEADLINE_TARGETS = [
    # (family, model, arch_substr)  -- match the row to include
    ('Residual MLP',        '--',                                'Residual MLP (Park puzzle, n=48)'),
    # GPT-2 family
    ('Transformer MLP',     'gpt2-xl',                           'GPT-2-xl MLP'),
    ('Attention V<->O',     'gpt2-xl',                           'GPT-2-xl Attn V/O'),
    ('Attention Q<->K',     'gpt2-xl',                           'GPT-2-xl Attn Q/K'),
    # ResNet family
    ('BasicBlock',          'resnet34/layer3',                   'ResNet-34 BasicBlock'),
    ('Bottleneck',          'resnet50/layer3',                   'ResNet-50 Bottleneck'),
    ('Bottleneck',          'resnet101/layer3',                  'ResNet-101 Bottleneck'),
    ('Bottleneck',          'resnet152/layer3',                  'ResNet-152 Bottleneck'),
    # Modern Vision
    ('ConvNeXt CNBlock',    'convnext_tiny/stage3',              'ConvNeXt-T stage3'),
    ('ViT MLP',             'vit_b_16',                          'ViT-B/16 MLP'),
    ('ViT Attn V<->O',      'vit_b_16',                          'ViT-B/16 Attn V/O'),
    ('ViT Attn Q<->K',      'vit_b_16',                          'ViT-B/16 Attn Q/K'),
]

# Build a row lookup
lookup = {}
for r in rows:
    if r.get('is_random_baseline'):
        continue
    key = (r['arch'], r['model'])
    lookup[key] = r

# Hardcode the Park puzzle row (not in any JSON since it's the original)
park_row = {
    'family': 'Residual MLP (no norm)', 'model': "Park's puzzle",
    'arch': 'Residual MLP', 'product': r'W_{\mathrm{out}} W_{\mathrm{in}}',
    'n': 48, 'pair_acc': 1.0, 'auc': 1.000, 'sep': 1.18,
    'neg_trace': 1.0,
}

LATEX_LINES = []
LATEX_LINES.append(r"\begin{tabular}{lllcccc}")
LATEX_LINES.append(r"\toprule")
LATEX_LINES.append(r"Model / stage & Branch type & Branch product & $n$ & Pair acc & AUC & Sep \\")
LATEX_LINES.append(r"\midrule")
LATEX_LINES.append(rf"Park's puzzle (Jane St.)        & Residual MLP        & $W_{{\mathrm{{out}}}} W_{{\mathrm{{in}}}}$ & 48 & \textbf{{100\%}} & 1.000 & $+1.18$ \\")

# GPT-2 family
for k, label in [(('Transformer MLP', 'gpt2'),         'GPT-2 (124M)'),
                 (('Transformer MLP', 'gpt2-medium'),  'GPT-2 medium (355M)'),
                 (('Transformer MLP', 'gpt2-large'),   'GPT-2 large (774M)'),
                 (('Transformer MLP', 'gpt2-xl'),      'GPT-2 xl (1.5B)')]:
    r = lookup.get(k)
    if r is None:
        continue
    sep_str = f"${r['sep']:+.3f}$" if r['sep'] is not None else '--'
    auc_str = f"{r['auc']:.3f}" if r['auc'] is not None else '--'
    LATEX_LINES.append(rf"{label:36s} & Transformer MLP     & $W_2 W_1$                         & {r['n']:>2d} & \textbf{{{r['pair_acc']:.0%}}} & {auc_str} & {sep_str} \\")

for k, label in [(('Attention V<->O', 'gpt2-xl'),      'GPT-2 xl Attn V/O'),
                 (('Attention Q<->K', 'gpt2-xl'),      'GPT-2 xl Attn Q/K')]:
    r = lookup.get(k)
    if r is None:
        continue
    arch_label = 'Attn V/O' if 'V<->O' in k[0] else 'Attn Q/K'
    prod = '$W_O W_V$' if 'V<->O' in k[0] else r'$W_Q W_K^\top$'
    sep_str = f"${r['sep']:+.3f}$" if r['sep'] is not None else '--'
    auc_str = f"{r['auc']:.3f}" if r['auc'] is not None else '--'
    LATEX_LINES.append(rf"{label:36s} & {arch_label:18s} & {prod:33s} & {r['n']:>2d} & \textbf{{{r['pair_acc']:.0%}}} & {auc_str} & {sep_str} \\")

# ResNet family
for k, label in [(('BasicBlock', 'resnet34/layer3'),     'ResNet-34 / layer3'),
                 (('Bottleneck', 'resnet50/layer3'),     'ResNet-50 / layer3'),
                 (('Bottleneck', 'resnet101/layer3'),    'ResNet-101 / layer3'),
                 (('Bottleneck', 'resnet152/layer3'),    'ResNet-152 / layer3')]:
    r = lookup.get(k)
    if r is None:
        continue
    arch_label = 'BasicBlock' if k[0] == 'BasicBlock' else 'Bottleneck'
    prod = '$W_2 W_1$' if k[0] == 'BasicBlock' else '$W_3 W_2 W_1$'
    sep_str = f"${r['sep']:+.3f}$" if r['sep'] is not None else '--'
    auc_str = f"{r['auc']:.3f}" if r['auc'] is not None else '--'
    LATEX_LINES.append(rf"{label:36s} & {arch_label:18s} & {prod:33s} & {r['n']:>2d} & \textbf{{{r['pair_acc']:.0%}}} & {auc_str} & {sep_str} \\")

# Modern vision
for k, label in [(('ConvNeXt CNBlock', 'convnext_tiny/stage3'), 'ConvNeXt-T / stage3'),
                 (('ViT MLP',           'vit_b_16'),             'ViT-B/16 MLP'),
                 (('ViT Attn V<->O',    'vit_b_16'),             'ViT-B/16 Attn V/O'),
                 (('ViT Attn Q<->K',    'vit_b_16'),             'ViT-B/16 Attn Q/K')]:
    r = lookup.get(k)
    if r is None:
        continue
    arch_short = {'ConvNeXt CNBlock': 'ConvNeXt MLP',
                  'ViT MLP':          'ViT MLP',
                  'ViT Attn V<->O':   'Attn V/O',
                  'ViT Attn Q<->K':   'Attn Q/K'}.get(k[0])
    prod = {'ConvNeXt CNBlock': '$W_2 W_1$',
            'ViT MLP':          '$W_2 W_1$',
            'ViT Attn V<->O':   '$W_O W_V$',
            'ViT Attn Q<->K':   r'$W_Q W_K^\top$'}.get(k[0])
    sep_str = f"${r['sep']:+.3f}$" if r['sep'] is not None else '--'
    auc_str = f"{r['auc']:.3f}" if r['auc'] is not None else '--'
    LATEX_LINES.append(rf"{label:36s} & {arch_short:18s} & {prod:33s} & {r['n']:>2d} & \textbf{{{r['pair_acc']:.0%}}} & {auc_str} & {sep_str} \\")

LATEX_LINES.append(r"\bottomrule")
LATEX_LINES.append(r"\end{tabular}")

latex = '\n'.join(LATEX_LINES)
with open('results/architecture_aware_master_table.tex', 'w') as f:
    f.write(latex)
print()
print('=' * 110)
print('LaTeX table (saved to results/architecture_aware_master_table.tex):')
print('=' * 110)
print(latex)
