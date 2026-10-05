"""Évaluation (sections 4.1 à 4.4) : erreurs L² relative et L∞ aux temps des
tables 3-4 et sur tout l'intervalle de test, échantillons hors distribution,
autres résolutions, RK2 et ablation de la perte de consistance.

    python evaluate.py --pde burgers
    python evaluate.py --pde advection

Chaque modèle absent de checkpoints/ est ignoré.
"""
import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
import data as D
import flux_fno as FF
from models import build
from train import run_name


# ------------------------------------------------------------------ chargement
def load(pde, name, rk2=False, lam=C.LAMBDA_CONSI):
    f = os.path.join(C.CKPT_DIR, run_name(pde, name, rk2, lam) + ".pkl")
    if not os.path.exists(f):
        return None
    with open(f, "rb") as fh:
        params = pickle.load(fh)["params"]
    return build(name, use_grid=name != "flux_fno"), params


# ------------------------------------------------------------------ inférence
def rollout(step, u0, n_save, every):
    "Applique `step` n_save*every fois ; renvoie (..., n_save+1, N)."
    def inner(u, _):
        return step(u), None

    def outer(u, _):
        u, _ = jax.lax.scan(inner, u, None, length=every)
        return u, u

    _, us = jax.lax.scan(outer, u0, None, length=n_save)
    return jnp.concatenate([u0[None], us], 0).swapaxes(0, -2)


def predict(pde, name, m, u0, T, every, rk2=False):
    """Prédit la trajectoire sur [0, T], un instantané tous les `every` pas de
    temps du jeu d'entraînement (Δt = dt_over_dx · Δx, à toute résolution)."""
    model, params = m
    n = u0.shape[-1]
    dx, dt = 1.0 / n, C.PDES[pde]["dt_over_dx"] / n
    n_steps = D.n_steps(pde, T, n)
    u0 = jnp.asarray(u0, jnp.float32)

    if name == "flux_fno":
        f = FF.rk2_step if rk2 else FF.euler_step
        step = jax.jit(lambda u: f(model, params, u, dt, dx))
        return np.asarray(rollout(step, u0, n_steps // every, every))

    if name in ("fno1d", "fno1d_heavy"):
        step = jax.jit(lambda u: model.apply(params, u[..., None])[..., 0])
        return np.asarray(rollout(step, u0, n_steps // every, every))

    if name == "fno2d":
        # Le 2D FNO prend en entrée le premier bloc de Tb pas (donné par la référence).
        Tb = C.FNO2D_BLOCK[pde]
        first = D.make_trajectories(pde, np.asarray(u0, np.float64), Tb * dt)[..., :Tb, :]
        n_blocks = -(-(n_steps + 1) // Tb)
        f = jax.jit(lambda b: model.apply(params, b[..., None])[..., 0])
        blocks = [jnp.asarray(first, jnp.float32)]
        for _ in range(n_blocks - 1):
            blocks.append(f(blocks[-1]))
        return np.asarray(jnp.concatenate(blocks, -2)[..., :n_steps + 1:every, :])

    if name == "fno_snap":
        # FNO(snap) : u0 -> u(T_snap), appliqué de façon répétée.
        f = jax.jit(lambda u: model.apply(params, u[..., None])[..., 0])
        Ts = C.PDES[pde]["T_train"]
        out, u = [u0], u0
        for _ in range(int(round(T / Ts))):
            u = f(u)
            out.append(u)
        return np.stack([np.asarray(o) for o in out], -2)
    raise ValueError(name)


# ------------------------------------------------------------------ métriques
def errors(pred, ref):
    "Erreur L² relative et L∞, moyennées sur les échantillons. (..., N)."
    l2 = np.linalg.norm(pred - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
    linf = np.abs(pred - ref).max(-1)
    return l2.mean(0), linf.mean(0)


def snap_errors(pred, ref, idx, times, T_snap):
    """FNO(snap) ne donne des instantanés qu'aux multiples de T_snap : erreurs
    à ces temps-là, NaN ailleurs. Renvoie (L², L∞, instantanés de l'éch. 0)."""
    ok = np.array([abs(t / T_snap - round(t / T_snap)) < 1e-3 for t in times])
    k = [int(round(t / T_snap)) for t in times]
    l2, li = errors(pred[:, k], ref[:, idx])
    return np.where(ok, l2, np.nan), np.where(ok, li, np.nan), np.where(ok[:, None], pred[0, k], np.nan)


def table(rows, times):
    head = f"{'modèle':<22}" + "".join(f"{'t=%.2f' % t:>22}" for t in times) + f"{'tout [0,T]':>22}"
    print(head)
    print("-" * len(head))
    for name, (l2, li) in rows.items():
        print(f"{name:<22}" + "".join(f"{'(%.3g, %.3g)' % (a, b):>22}" for a, b in zip(l2, li)))


def plot_snapshots(path, x, ref, preds, times, title):
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 3.2), squeeze=False)
    for ax, t, k in zip(axes[0], times, range(len(times))):
        ax.plot(x, ref[k], "k-", lw=2, label="référence")
        for name, p in preds.items():
            ax.plot(x, p[k], "--", lw=1.3, label=name)
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ expériences
def main_comparison(pde, models):
    "Section 4.1 (tables 3-4), 4.3 (table 5) et figures 3-6."
    P = C.PDES[pde]
    d = np.load(D.path(f"{pde}.npz"))
    ref, test_dt = d["test"], float(d["test_dt"])
    every = int(round(test_dt / float(d["dt"])))
    T = P["T_test"]
    idx = [int(round(t / test_dt)) for t in P["eval_times"]]
    times = [i * test_dt for i in idx]

    rows, snaps = {}, {}
    for label, (name, m, rk2) in models.items():
        t0 = time.time()
        pred = predict(pde, name, m, ref[:, 0], T, every, rk2)
        dt_inf = time.time() - t0
        if name == "fno_snap":
            l2, li, snaps[label] = snap_errors(pred, ref, idx, times, P["T_train"])
            rows[label] = (list(l2) + [np.nan], list(li) + [np.nan])
        else:
            e_t = errors(pred[:, idx], ref[:, idx])
            e_all = errors(pred[:, 1:].reshape(-1, ref.shape[-1]), ref[:, 1:].reshape(-1, ref.shape[-1]))
            rows[label] = (list(e_t[0]) + [e_all[0]], list(e_t[1]) + [e_all[1]])
            snaps[label] = pred[0, idx]
        print(f"  {label}: inférence {dt_inf:.1f}s")
    print(f"\n=== {pde} : (L² relative, L∞), moyenne sur le jeu de test ===")
    table(rows, times)
    plot_snapshots(os.path.join(C.FIG_DIR, f"{pde}_comparaison.png"), D.grid(C.N),
                   ref[0, idx], snaps, times, f"{pde} — comparaison des modèles")


def ood(pde, models):
    "Section 4.2 : échantillons hors distribution (figures 7-8)."
    P = C.PDES[pde]
    rng = np.random.default_rng(C.SEED + 1)
    T = P["T_test"]
    times = P["eval_times"]
    n = C.N
    dt = P["dt_over_dx"] / n
    cases = {"GRF(0.03)": D.sample_grf(rng, 10, n, C.GRF_LENGTH_OOD)}
    if pde == "advection":
        cases["triangle"] = D.triangle_pulse(n)[None]
    else:
        cases["marche"] = D.step_function(n)[None]

    rows = {}
    for case, u0 in cases.items():
        every = D.n_steps(pde, T, n) // 1280
        if pde == "burgers" and case == "marche":
            ref = np.stack([D.burgers_step_exact(k * every * dt, n) for k in range(1281)])[None]
        else:
            ref = D.make_trajectories(pde, u0, T, save_every=every)
        idx = [int(round(t / (every * dt))) for t in times]
        preds = {}
        for label, (name, m, rk2) in models.items():
            if name == "fno_snap":
                p = predict(pde, name, m, u0, T, every)
                l2, li, preds[label] = snap_errors(p, ref, idx, times, P["T_train"])
            else:
                p = predict(pde, name, m, u0, T, every, rk2)
                preds[label] = p[0, idx]
                l2, li = errors(p[:, idx], ref[:, idx])
            rows[f"{label} / {case}"] = (list(l2), list(li))
        plot_snapshots(os.path.join(C.FIG_DIR, f"{pde}_ood_{case}.png"), D.grid(n),
                       ref[0, idx], preds, times, f"{pde} — OOD : {case}")
    print(f"\n=== {pde} : hors distribution (L² relative, L∞) ===")
    table(rows, times)


def resolution(pde, models):
    "Section 4.2 : invariance en résolution (figures 9-10, table 6)."
    P = C.PDES[pde]
    rng = np.random.default_rng(C.SEED + 2)
    times = P["eval_times"]
    rows = {}
    for n in (128, 256, 512):
        u0 = D.sample_grf(rng, 10, n, C.GRF_LENGTH)
        dt = P["dt_over_dx"] / n
        every = max(1, D.n_steps(pde, P["T_test"], n) // 1280)
        ref = D.make_trajectories(pde, u0, P["T_test"], save_every=every)
        idx = [int(round(t / (every * dt))) for t in times]
        preds = {}
        for label, (name, m, rk2) in models.items():
            if name != "flux_fno":
                continue
            p = predict(pde, name, m, u0, P["T_test"], every, rk2)
            preds[label] = p[0, idx]
            rows[f"{label} / N={n}"] = errors(p[:, idx], ref[:, idx])
        if n != C.N:
            plot_snapshots(os.path.join(C.FIG_DIR, f"{pde}_resolution_{n}.png"), D.grid(n),
                           ref[0, idx], preds, times, f"{pde} — résolution N = {n}")
    print(f"\n=== {pde} : résolutions (L² relative, L∞) ===")
    table(rows, times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", choices=list(C.PDES), required=True)
    args = ap.parse_args()
    os.makedirs(C.FIG_DIR, exist_ok=True)
    pde = args.pde

    candidates = {
        "Flux FNO": ("flux_fno", False, C.LAMBDA_CONSI),
        "Flux FNO + RK2": ("flux_fno", True, C.LAMBDA_CONSI),
        "Flux FNO sans L_consi": ("flux_fno", False, 0.0),
        "1D FNO": ("fno1d", False, None),
        "1D FNO (heavy)": ("fno1d_heavy", False, None),
        "2D FNO": ("fno2d", False, None),
        "FNO(snap)": ("fno_snap", False, None),
    }
    models = {}
    for label, (name, rk2, lam) in candidates.items():
        m = load(pde, name, rk2, lam if lam is not None else C.LAMBDA_CONSI)
        if m is not None:
            models[label] = (name, m, rk2)
    if not models:
        raise SystemExit("aucun checkpoint trouvé : lancer train.py d'abord")
    print("modèles :", ", ".join(models))

    main_comparison(pde, models)
    ood(pde, {k: v for k, v in models.items() if v[0] in ("flux_fno", "fno2d", "fno_snap")})
    resolution(pde, models)


if __name__ == "__main__":
    main()
