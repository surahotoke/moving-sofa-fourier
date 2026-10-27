"""
動くソファ問題:ハマーズリーの動きをフーリエ級数で一般化して面積を最適化する

ソファに固定した視点で、廊下の内側のかど(以下「かど」)が中心 O の周りを回る。
  廊下の回転角 phi in [0, pi/2],  theta = phi - pi/4
  かどの位置 c = y(theta) * (cos(2phi + d(theta)), -sin(2phi + d(theta)))
  y(theta) = r0 + sum_k a_k cos(2k theta)      (距離)
  d(theta) = sum_k b_k sin(2k theta)           (向きのずれ)
ソファ = 動いていくすべての廊下の位置の共通部分。

  使い方: python moving_sofa_fourier.py --mode both --K 3
    --mode both  : 距離と角度の両方を動かす(既定)。曲がり始めのかどは原点の真横に置く
    --mode y     : 距離だけを動かす(δ = 0)
    --mode delta : 角度だけを動かす(y = 2/π)。曲がり始めのかどは原点の真横に置く
    K 項の数え方: 距離側は r0 を含めて K 個、角度側は K 個

  全部0 (r0=0)        -> 半円        pi/2      ~ 1.5708
  r0 = 2/pi のみ      -> ハマーズリー pi/2+2/pi ~ 2.2074
  a_k, b_k も最適化   -> ガーバーに接近          4項ずつで 2.219527 (ガーバー 2.219532)

"""
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely import affinity

M = 8.0  # 廊下の腕の長さ(無限の代わり)
# 廊下(内側のかどが原点、外側のかどが (1,-1))
H = Polygon([(-M, -1), (1, -1), (1, 0), (-M, 0)])   # 横の腕
V = Polygon([(0, -1), (1, -1), (1, M), (0, M)])      # 縦の腕
L = H.union(V)


def place(poly, phi, c):
    """廊下の形をソファ視点に置く: s = c + R(-phi) h"""
    p = affinity.rotate(poly, -phi, origin=(0, 0), use_radians=True)
    return affinity.translate(p, c[0], c[1])


def sofa(r0, a=(), b=(), n=400):
    """パラメータから最大のソファ(shapely の図形)を作る"""
    def corner(phi):
        t = phi - np.pi / 4
        r = r0 + sum(ak * np.cos(2 * (k + 1) * t) for k, ak in enumerate(a))
        d = sum(bk * np.sin(2 * (k + 1) * t) for k, bk in enumerate(b))
        return (r * np.cos(2 * phi + d), -r * np.sin(2 * phi + d))

    S = place(H, 0.0, corner(0.0))                          # 横の腕から入ってくる
    S = S.intersection(place(V, np.pi / 2, corner(np.pi / 2)))  # 縦の腕へ出ていく
    for phi in np.linspace(0, np.pi / 2, n):                # 回転中
        S = S.intersection(place(L, phi, corner(phi)))
        if S.is_empty:
            break
    return S


def area(r0, a=(), b=(), n=400):
    """角度の区切りによる誤差を 2A(2n) - A(n) で補正した面積"""
    return 2 * sofa(r0, a, b, 2 * n).area - sofa(r0, a, b, n).area


def b1_from_rest(b_rest):
    """曲がり始めのかどを、ハマーズリーと同じく原点の真横に置く条件 δ(φ=0) = 0 から b_1 を決める。
    δ(−45°) = −b1 + b3 − b5 + b7 − … = 0  より  b1 = b3 − b5 + b7 − …
    これは設計上の選択で、有限項では自由度が一つ減る制約になるが、項を増やせば影響は消える。"""
    # b_rest = [b_2, b_3, ..., b_K]  (b_k は index k-2)
    return sum(((-1) ** ((k - 3) // 2)) * b_rest[k - 2] for k in range(3, len(b_rest) + 2, 2))


def unpack(x, K, mode):
    """係数の並び x を (r0, a, b) に分ける"""
    if mode == "both":   # x = [r0, a_1..a_{K-1}, b_2..b_K]  (b_1 は条件から決まる)
        b_rest = x[K:]
        return x[0], x[1:K], np.r_[b1_from_rest(b_rest), b_rest]
    if mode == "y":      # 距離だけ: x = [r0, a_1..a_{K-1}], δ = 0
        return x[0], x[1:], ()
    if mode == "delta":  # 角度だけ: x = [b_2..b_K], y = 2/π  (b_1 は条件から決まる)
        return 2 / np.pi, (), np.r_[b1_from_rest(x), x]
    raise ValueError(mode)


def start(mode):
    """K = 1 の出発点"""
    return {"both": np.r_[2 / np.pi], "y": np.r_[2 / np.pi], "delta": np.zeros(0)}[mode]


def grow(x, K, mode):
    """K-1 項の解に 0 を足して K 項の出発点にする"""
    if mode == "both":
        return np.r_[x[:K - 1], 0.0, x[K - 1:], 0.0]
    return np.r_[x, 0.0]


def optimize(K, mode, x0, n=100):
    """全部の係数を同時に最適化する。
    目的関数には補正済みの面積 2A(2n) - A(n) を使う。補正前の面積は形ごとに違う量だけ
    大きめに出るため、それを最大化すると本当の最適解からずれた所で止まってしまう。"""
    if np.size(x0) == 0:   # 動かす係数がない(角度だけの1項はハマーズリーそのもの)
        return x0, area(*unpack(x0, K, mode), n=960)
    res = minimize(lambda x: -area(*unpack(x, K, mode), n=n), x0, method="Powell",
                   options={"xtol": 1e-5, "ftol": 1e-10, "maxfev": 3000})
    return res.x, area(*unpack(res.x, K, mode), n=960)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="動くソファ問題:三角関数による動きの最適化")
    p.add_argument("--mode", choices=["both", "y", "delta"], default="both",
                   help="both: 距離と角度の両方 / y: 距離だけ (δ=0) / delta: 角度だけ (y=2/π)")
    p.add_argument("--K", type=int, default=3, help="項数の上限 (既定 3)")
    args = p.parse_args()

    print(f"半円        {area(0.0):.5f}   (厳密値 {np.pi/2:.5f})")
    print(f"ハマーズリー {area(2/np.pi):.5f}   (厳密値 {np.pi/2 + 2/np.pi:.5f})")
    print(f"--- mode = {args.mode} ---")
    x = start(args.mode)
    for K in range(1, args.K + 1):
        if K > 1:
            x = grow(x, K, args.mode)
        x, A = optimize(K, args.mode, x)
        r0, a, b = unpack(x, K, args.mode)
        print(f"K={K}  面積 {A:.6f}  r0={r0:.4f}  a={np.round(a, 4)}  b={np.round(b, 4)}")
    print("ガーバー     2.219532 (参考値)")
