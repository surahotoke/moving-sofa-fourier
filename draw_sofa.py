"""
係数 r0, a, b からソファの形を描いて PNG に保存するスクリプト。
moving_sofa_fourier.py と同じフォルダで実行してください。

使い方(moving_sofa_fourier.py の出力をそのまま写せます):
  python3 draw_sofa.py --r0 0.6473 --a -0.0157 0.033 --b 0.012 -0.0752 0.012
  python3 draw_sofa.py --r0 0.6473 --a -0.0157 0.033 --b 0.012 -0.0752 0.012 --phi 15 --out mysofa.png

  --phi を付けると、その曲がり具合(度)の廊下も重ねて描きます。
  何も付けなければハマーズリーのソファ(r0 = 2/π)を描きます。
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from shapely import affinity
from shapely.geometry import box
from moving_sofa_fourier import sofa, area, place, L

from matplotlib import font_manager
_have = {f.name for f in font_manager.fontManager.ttflist}
for _name in ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP", "Yu Gothic", "Meiryo"]:
    if _name in _have:
        plt.rcParams["font.family"] = _name
        break

p = argparse.ArgumentParser(description="係数からソファの形を描く")
p.add_argument("--r0", type=float, default=2 / np.pi, help="距離の定数項(既定 2/π)")
p.add_argument("--a", type=float, nargs="*", default=[], help="距離の係数 a_1 a_2 …(cos 2θ, cos 4θ, …)")
p.add_argument("--b", type=float, nargs="*", default=[], help="角度の係数 b_1 b_2 …(sin 2θ, sin 4θ, …)")
p.add_argument("--phi", type=float, default=None, help="重ねて描く廊下の曲がり具合(度、0〜90)")
p.add_argument("--out", default="sofa.png", help="保存するファイル名(既定 sofa.png)")
args = p.parse_args()

S = sofa(args.r0, args.a, args.b, n=1000)
A = area(args.r0, args.a, args.b, n=960)
print(f"面積 {A:.6f}  (ハマーズリー 2.207416 / ガーバー 2.219532)")

# 記事の図と同じ向き(まっすぐな辺が下)にするため 180 度回す
rot = lambda g: affinity.rotate(g, 180, origin=(0, 0))
S = rot(S)

fig, ax = plt.subplots(figsize=(8, 5))
if args.phi is not None:
    phi = np.radians(args.phi)
    t = phi - np.pi / 4
    y = args.r0 + sum(ak * np.cos(2 * (k + 1) * t) for k, ak in enumerate(args.a))
    d = sum(bk * np.sin(2 * (k + 1) * t) for k, bk in enumerate(args.b))
    c = (y * np.cos(2 * phi + d), -y * np.sin(2 * phi + d))
    hall = rot(place(L, phi, c)).intersection(box(*S.buffer(1.2).bounds))
    for g in ([hall] if hall.geom_type == "Polygon" else hall.geoms):
        ax.add_patch(MPoly(np.array(g.exterior.coords), closed=True, fc="#cfe3f5", ec="#4a90c8", lw=1.5, alpha=0.6))
for g in ([S] if S.geom_type == "Polygon" else S.geoms):
    ax.add_patch(MPoly(np.array(g.exterior.coords), closed=True, fc="#f4b6b6", ec="#d62728", lw=1.5, alpha=0.85))
minx, miny, maxx, maxy = S.buffer(0.3 if args.phi is None else 1.0).bounds
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_aspect("equal"); ax.axis("off"); ax.invert_xaxis()
title = f"面積 {A:.6f}"
if args.phi is not None:
    title += f"(φ = {args.phi:g}° の廊下つき)"
ax.set_title(title)
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"{args.out} に保存しました")
