#!/usr/bin/env python3
"""Colorize a 3D scan of a cup-marked / engraved stone to make the markings visible.

Each vertex is colored by its *local relief*: how far it sits below (hollow) or
above (ridge) a smoothed version of the surface. Smoothing is done by heat
diffusion on the mesh itself (cotangent Laplacian), so it follows the stone's
shape, is independent of how the scan is oriented, and does not bleed across
gaps between separate rocks.

Two scale parameters choose which feature size is highlighted:
  --scale  S   features up to about this size are kept (e.g. cup diameter)
  --fine   s   features smaller than this are suppressed (natural roughness)
Both are in mesh units (the Nr38 scan appears to be in metres).

Only numpy and scipy are required.

Example:
  python colorize_stone.py Nr38.obj Nr38_colored.obj --scale 0.15 --fine 0.02
"""
import argparse
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------- OBJ I/O
def read_obj(path):
    """Return vertices (n,3) and triangle vertex indices (m,3)."""
    verts, faces = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                    faces.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    F = np.where(F > 0, F - 1, F + len(V))  # handle negative (relative) indices
    return V, F


def write_obj(src, dst, colors, rotation=None, center=None):
    """Copy src to dst, replacing per-vertex colors (and optionally rotating)."""
    i = 0
    with open(src, "r", errors="replace") as fi, open(dst, "w") as fo:
        for line in fi:
            if line.startswith("v "):
                p = np.array(line.split()[1:4], dtype=np.float64)
                if rotation is not None:
                    p = rotation @ (p - center)
                r, g, b = colors[i]
                fo.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {r:.6f} {g:.6f} {b:.6f}\n")
                i += 1
            elif line.startswith("vn ") and rotation is not None:
                n = rotation @ np.array(line.split()[1:4], dtype=np.float64)
                fo.write(f"vn {n[0]:.7f} {n[1]:.7f} {n[2]:.7f}\n")
            elif line.startswith("mtllib ") or line.startswith("usemtl "):
                continue  # drop texture so viewers show the vertex colors
            else:
                fo.write(line)


# ---------------------------------------------------------------- geometry
def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # area weighted
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)


def laplacian(V, F):
    """Cotangent stiffness matrix L (positive semi-definite) and lumped mass M."""
    n = len(V)
    I, J, W = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = V[F[:, b]] - V[F[:, a]]
        w = V[F[:, c]] - V[F[:, a]]
        cot = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-20)
        cot = np.clip(cot, -1e3, 1e3)  # angle at a weights the opposite edge b-c
        I += [F[:, b], F[:, c]]
        J += [F[:, c], F[:, b]]
        W += [0.5 * cot, 0.5 * cot]
    I, J, W = np.concatenate(I), np.concatenate(J), np.concatenate(W)
    W = np.maximum(W, 0.0)  # drop negative weights: keeps the smoothing well behaved
    A = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    m = np.zeros(n)
    for k in range(3):
        np.add.at(m, F[:, k], area / 3.0)
    m = np.maximum(m, 1e-12 * max(m.max(), 1e-20))  # isolated vertices
    return L, sp.diags(m)


class Smoother:
    """Heat diffusion: solve (M + t L) x = M x0, with t = sigma^2 / 2."""

    def __init__(self, V, F):
        self.L, self.M = laplacian(V, F)

    def __call__(self, X, sigma):
        if sigma <= 0:
            return X.copy()
        A = (self.M + (sigma ** 2 / 2.0) * self.L).tocsc()
        solve = spla.factorized(A)
        B = self.M @ X
        return np.column_stack([solve(B[:, k]) for k in range(X.shape[1])])


def local_relief(V, F, scale, fine=0.0, S=None):
    """Signed distance of the fine-smoothed surface above the coarse-smoothed one."""
    S = S or Smoother(V, F)
    coarse = S(V, scale / 2.0)  # sigma ~ half the feature size
    detail = S(V, fine / 2.0) if fine > 0 else V
    N = vertex_normals(coarse, F)
    return np.einsum("ij,ij->i", detail - coarse, N)


def roughness(V, F, smoother, radius):
    """RMS of very fine (few-mm) surface detail, averaged over `radius`.

    Bare rock is smooth at this scale, while grass, gravel and scan noise in
    the surrounding terrain are rough, so this separates stone from terrain."""
    sm = smoother(V, 0.02)
    rf = np.einsum("ij,ij->i", V - sm, vertex_normals(sm, F))
    return np.sqrt(np.maximum(smoother(rf[:, None] ** 2, radius)[:, 0], 0.0))


def otsu_threshold(x, bins=256):
    h, b = np.histogram(x, bins)
    c = 0.5 * (b[:-1] + b[1:])
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    var = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w) + 1e-9)
    return c[np.argmax(var)]


def stone_weight(rough, threshold=None):
    """1 on stone, fading to 0 on rough terrain. Threshold auto (Otsu on log) if None."""
    lr = np.log10(np.maximum(rough, 1e-9))
    t = otsu_threshold(lr[rough > 1e-6]) if threshold is None else np.log10(threshold)
    return np.clip((t + 0.1 - lr) / 0.2, 0.0, 1.0), 10 ** t


def level_rotation(V, F):
    """Rotation that makes the best-fit plane horizontal, surface facing +Z."""
    c = V.mean(axis=0)
    _, vecs = np.linalg.eigh((V - c).T @ (V - c))
    up = vecs[:, 0]
    if vertex_normals(V, F).sum(axis=0) @ up < 0:
        up = -up
    x = vecs[:, 2]
    y = np.cross(up, x)
    return np.vstack([x, y, up]), c


# ---------------------------------------------------------------- colors
COLORMAPS = {
    # hollows -> ... -> ridges
    "diverging": [(0.05, 0.10, 0.45), (0.15, 0.45, 0.85), (0.92, 0.92, 0.88),
                  (0.95, 0.60, 0.20), (0.60, 0.10, 0.05)],
    "gray": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    "hollows": [(0.10, 0.05, 0.30), (0.80, 0.20, 0.10), (0.95, 0.85, 0.50),
                (0.85, 0.85, 0.85), (0.85, 0.85, 0.85)],
}


def apply_colormap(t, name):
    stops = np.asarray(COLORMAPS[name])
    x = np.linspace(0.0, 1.0, len(stops))
    return np.column_stack([np.interp(t, x, stops[:, k]) for k in range(3)])


def relief_to_colors(r, cmap, clip, gamma, weight=None, terrain_gray=0.55):
    ref = r if weight is None or (weight > 0.5).sum() < 100 else r[weight > 0.5]
    lo, hi = np.percentile(ref, [clip, 100.0 - clip])
    lim = max(-lo, hi, 1e-12)  # symmetric, so zero relief maps to the middle color
    t = np.clip(r / lim, -1.0, 1.0)
    t = np.sign(t) * np.abs(t) ** gamma
    rgb = apply_colormap(0.5 + 0.5 * t, cmap)
    if weight is not None:  # terrain: faint gray version of the relief
        gray = terrain_gray + 0.25 * t[:, None]
        rgb = weight[:, None] * rgb + (1.0 - weight[:, None]) * np.repeat(gray, 3, axis=1)
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="source .obj")
    p.add_argument("output", help="colored .obj to write")
    p.add_argument("--scale", type=float, default=None,
                   help="largest feature size to highlight, in mesh units (default: 1.2%% of the mesh diagonal)")
    p.add_argument("--fine", type=float, default=0.0,
                   help="suppress features smaller than this (natural roughness); 0 = keep all detail")
    p.add_argument("--cmap", choices=sorted(COLORMAPS), default="diverging")
    p.add_argument("--clip", type=float, default=1.0,
                   help="percent of extreme values to saturate at each end (contrast stretch)")
    p.add_argument("--gamma", type=float, default=0.7, help="<1 boosts faint features")
    p.add_argument("--terrain", default="auto",
                   help="gray out rough terrain (grass, gravel): 'auto' picks a roughness threshold, "
                        "'off' colors everything, or give the threshold in mesh units (e.g. 0.002)")
    p.add_argument("--level", action="store_true",
                   help="also rotate the output so the stone lies flat (best-fit plane horizontal)")
    p.add_argument("--save-relief", metavar="NPY", help="save raw per-vertex relief values")
    a = p.parse_args(argv)

    print(f"reading {a.input} ...", file=sys.stderr)
    V, F = read_obj(a.input)
    diag = np.linalg.norm(V.max(0) - V.min(0))
    scale = a.scale if a.scale else 0.012 * diag
    print(f"{len(V)} vertices, {len(F)} faces, size {diag:.3g}; scale={scale:.3g} fine={a.fine:.3g}",
          file=sys.stderr)

    S = Smoother(V, F)
    r = local_relief(V, F, scale, a.fine, S)
    if a.save_relief:
        np.save(a.save_relief, r)
    weight = None
    if a.terrain != "off":
        thr = None if a.terrain == "auto" else float(a.terrain)
        weight, thr = stone_weight(roughness(V, F, S, max(scale, 0.1)), thr)
        print(f"terrain roughness threshold {thr:.3g}; {100 * (weight > 0.5).mean():.0f}% of vertices treated as stone",
              file=sys.stderr)
    colors = relief_to_colors(r, a.cmap, a.clip, a.gamma, weight)

    rot = ctr = None
    if a.level:
        rot, ctr = level_rotation(V, F)
    write_obj(a.input, a.output, colors, rot, ctr)
    print(f"wrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
