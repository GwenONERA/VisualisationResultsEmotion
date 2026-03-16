import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import textwrap, re, argparse
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  ARGUMENTS CLI
# ═══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description="Graphe en étoile : phrases reliées à une émotion centrale.")

parser.add_argument(
    '-n', '--max-texts',
    type=int,
    default=None,
    metavar='N',
    help="Nombre maximal de textes (phrases) à intégrer dans le graphe. "
         "Par défaut tous les textes trouvés sont affichés.")

parser.add_argument(
    '-e', '--emotion',
    type=str,
    default='colere',
    metavar='MOT',
    help="Mot-émotion servant de nœud central "
         "(doit correspondre à une valeur de la colonne "
         "'Categories of the emotion lemmas'). "
         "Par défaut : 'colere'.")

args = parser.parse_args()

MAX_TEXTS    = args.max_texts
EMOTION_WORD = args.emotion
EMOTION_LABEL = EMOTION_WORD.upper()

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CHARGEMENT & FILTRAGE
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_excel('/content/output_realigned.xlsx')

mask = df['Categories of the emotion lemmas'].fillna('').str.contains(
    EMOTION_WORD, regex=False)
df_emotion = df[mask].copy().reset_index(drop=True)

records = []
for _, row in df_emotion.iterrows():
    if pd.isna(row['sentence']) or pd.isna(row['Tokens having emotion']):
        continue
    cats = [c.strip() for c in str(row['Categories of the emotion lemmas']).split(',')]
    toks = [t.strip() for t in str(row['Tokens having emotion']).split(',')]
    idx  = next((i for i, c in enumerate(cats) if c == EMOTION_WORD), 0)
    records.append({
        'sentence': str(row['sentence']).strip(),
        'token':    toks[min(idx, len(toks) - 1)],
    })

data = (pd.DataFrame(records)
          .loc[lambda d: d['sentence'].ne('') & d['sentence'].ne('nan')]
          .drop_duplicates(subset=['sentence', 'token'])
          .reset_index(drop=True))

if MAX_TEXTS is not None and MAX_TEXTS > 0:
    data = data.head(MAX_TEXTS).reset_index(drop=True)

N = len(data)
print(f"Phrases à afficher : {N}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PHRASE AVEC TOKEN SURLIGNÉ
# ═══════════════════════════════════════════════════════════════════════════════
def make_highlighted_box(sentence, token, fontsize=28, wrap_width=34):
    try:
        pat = re.compile(r'(' + re.escape(token) + r')', re.IGNORECASE)
    except re.error:
        pat = None

    wrapped = textwrap.fill(sentence, width=wrap_width)
    lines   = wrapped.split('\n')
    line_areas, found = [], False

    for line in lines:
        m = pat.search(line) if (pat and not found) else None
        if m:
            before, tok_str, after = line[:m.start()], line[m.start():m.end()], line[m.end():]
            parts = []
            if before:
                parts.append(TextArea(before,
                    textprops=dict(fontsize=fontsize, color='#2C3E50',
                                   fontfamily='serif')))
            parts.append(TextArea(tok_str,
                textprops=dict(fontsize=fontsize + 4,
                               color='#B71C1C', fontweight='bold',
                               fontfamily='serif',
                               bbox=dict(facecolor='#FFCDD2',
                                         edgecolor='#E53935',
                                         boxstyle='round,pad=0.25',
                                         lw=1.6))))
            if after:
                parts.append(TextArea(after,
                    textprops=dict(fontsize=fontsize, color='#2C3E50',
                                   fontfamily='serif')))
            line_areas.append(HPacker(children=parts,
                                      align='baseline', pad=0, sep=1))
            found = True
        else:
            line_areas.append(TextArea(line,
                textprops=dict(fontsize=fontsize, color='#2C3E50',
                               fontfamily='serif')))

    if not found:
        line_areas.append(TextArea(f"  [→ {token}]",
            textprops=dict(fontsize=fontsize - 2, color='#B71C1C',
                           fontstyle='italic')))

    return VPacker(children=line_areas, align='left', pad=0, sep=5)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  GÉOMÉTRIE
# ═══════════════════════════════════════════════════════════════════════════════
R = max(10, N * 1.2)
FIGSIZE = max(36, min(90, int(2 * R + 20)))

angles = np.linspace(0, 2 * np.pi, N, endpoint=False) - np.pi / 2
jitter = np.array([1.2 * ((-1) ** i) for i in range(N)])

# ═══════════════════════════════════════════════════════════════════════════════
# 3b. CORRECTION CHEVAUCHEMENT POUR N == 10
#     Lorsque N == 10, les boîtes en haut (indices 4,5,6) et en bas (0,1,9)
#     se chevauchent. On décale verticalement les extrémités pour séparer.
# ═══════════════════════════════════════════════════════════════════════════════
vertical_offsets = {}
if N == 10:
    # Haut : indices 4 (top-right) et 6 (top-left) décalés vers le bas
    vertical_offsets[4] = -3.5
    vertical_offsets[6] = -3.5
    # Bas : indices 1 (bottom-right) et 9 (bottom-left) décalés vers le haut
    vertical_offsets[1] = +3.5
    vertical_offsets[9] = +3.5

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE),
                       subplot_kw={'aspect': 'equal'})
ax.axis('off')

ax.add_patch(plt.Circle((0, 0), R, fill=False,
                         ls=':', lw=.6, color='#D5D8DC', zorder=1))

# ── CENTRE : ÉMOTION ────────────────────────────────────────────────────────
CR = 1.5
ax.add_patch(plt.Circle((0, 0), CR,
                         color='#C62828', ec='#7F0000', lw=5, zorder=10))

ax.text(0, 0, EMOTION_LABEL, ha='center', va='center',
        fontsize=24, fontweight='bold', color='white', zorder=11)

# ── PHRASES ──────────────────────────────────────────────────────────────────
for i, (_, row) in enumerate(data.iterrows()):
    a   = angles[i]
    r_i = R + jitter[i]
    sx  = r_i * np.cos(a)
    sy  = r_i * np.sin(a)

    # Correction anti-chevauchement pour N == 10
    if i in vertical_offsets:
        sy += vertical_offsets[i]

    # Correction originale (gardée)
    if f"ça vous met en {EMOTION_WORD}" in row['sentence'].lower():
        sy -= 3.5

    packed = make_highlighted_box(row['sentence'], row['token'],
                                  fontsize=28, wrap_width=34)
    ab = AnnotationBbox(
        packed, (sx, sy), frameon=True,
        bboxprops=dict(boxstyle='round,pad=0.9',
                       fc='#FFF8E1', ec='#F9A825',
                       lw=1.5, alpha=0.95),
        pad=0.6, zorder=5)
    ax.add_artist(ab)

    ax.annotate('',
        xy=(0, 0), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle='-|>',
            color='#C62828',
            lw=3.0,
            shrinkA=80,
            shrinkB=25,
            connectionstyle='arc3,rad=0.03',
            alpha=0.55))

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
PAD = 16
ax.set_xlim(-R - PAD, R + PAD)
ax.set_ylim(-R - PAD, R + PAD)

plt.tight_layout(rect=[0, 0, 1, 1])

output_filename = f'{EMOTION_WORD}_direct_v3_lowres.png'
plt.savefig(output_filename,
            dpi=72,
            bbox_inches='tight',
            pad_inches=0.0,
            transparent=True)

plt.show()

print(f"\n✅  {N} phrases directement reliées à {EMOTION_LABEL}.")
