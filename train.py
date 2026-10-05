"""Entraînement du Flux FNO et des baselines FNO.

    python train.py --pde advection --model flux_fno
    python train.py --pde burgers   --model flux_fno [--rk2] [--lam 0]
    python train.py --pde burgers   --model fno1d | fno1d_heavy | fno2d | fno_snap

Comme dans le papier, chaque lot du Flux FNO et des 1D FNO est une suite de
pas de temps consécutifs d'une même fonction (pas d'instantanés tirés au
hasard) ; seul l'ordre des lots est mélangé à chaque epoch.
"""
import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import config as C
import flux_fno as FF
from data import path as data_path
from models import build


# ------------------------------------------------------------------ données
def one_step_pairs(U, B):
    """U : (m, T+1, N) -> X, Y : (m * (T // B), B, N), lots de pas consécutifs."""
    m, T1, n = U.shape
    nb = (T1 - 1) // B
    X = U[:, :nb * B].reshape(m * nb, B, n)
    Y = U[:, 1:nb * B + 1].reshape(m * nb, B, n)
    return X, Y


def block_pairs(U, Tb):
    "Pour le 2D FNO : bloc de Tb pas -> bloc suivant de Tb pas."
    m, T1, n = U.shape
    nb = T1 // Tb
    blocks = U[:, :nb * Tb].reshape(m, nb, Tb, n)
    return blocks[:, :-1].reshape(-1, Tb, n), blocks[:, 1:].reshape(-1, Tb, n)


def load_dataset(pde, name):
    if name == "fno_snap":
        d = np.load(data_path(f"{pde}_snap.npz"))
        return d["u0"], d["uT"], C.SNAP_BATCH, None
    d = np.load(data_path(f"{pde}.npz"))
    if name == "fno2d":
        X, Y = block_pairs(d["train"], C.FNO2D_BLOCK[pde])
        return X, Y, C.FNO2D_BATCH, float(d["dt"])
    X, Y = one_step_pairs(d["train"], C.PDES[pde]["batch_size"])
    return X, Y, 1, float(d["dt"])           # 1 « échantillon » = 1 lot consécutif


# ------------------------------------------------------------------ pertes
def make_loss(pde, name, model, dt, lam, rk2):
    if name == "flux_fno":
        return lambda p, x, y: FF.loss(model, p, pde, x, y, dt, lam, rk2)

    def loss(p, x, y):                        # FNO classique : régression directe
        r = (model.apply(p, x[..., None])[..., 0] - y) ** 2
        return jnp.sum(r), (jnp.mean(r), jnp.zeros(()))   # somme : cf. flux_fno.loss
    return loss


def make_optimizer(steps_per_epoch):
    # torch.optim.Adam(weight_decay) ajoute wd·θ au gradient avant Adam.
    sched = optax.exponential_decay(C.LR, C.STEP_SIZE * steps_per_epoch, C.GAMMA, staircase=True)
    return optax.chain(optax.add_decayed_weights(C.WEIGHT_DECAY), optax.scale_by_adam(),
                       optax.scale_by_learning_rate(sched))


def fit(loss_fn, params, X, Y, batch, epochs, key, log_every=10):
    S = X.shape[0]
    nb = S // batch
    opt = make_optimizer(nb)
    opt_state = opt.init(params)
    X, Y = jnp.asarray(X), jnp.asarray(Y)

    @jax.jit
    def epoch(params, opt_state, key, X, Y):
        idx = jax.random.permutation(key, S)[:nb * batch].reshape(nb, batch)

        def step(carry, i):
            params, opt_state = carry
            x, y = X[i], Y[i]
            if batch == 1:
                x, y = x[0], y[0]
            (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
            upd, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, upd), opt_state), (l, *aux)

        (params, opt_state), logs = jax.lax.scan(step, (params, opt_state), idx)
        return params, opt_state, jax.tree_util.tree_map(jnp.mean, logs)

    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        key, sub = jax.random.split(key)
        params, opt_state, (l, l_tm, l_c) = epoch(params, opt_state, sub, X, Y)
        hist.append((float(l), float(l_tm), float(l_c)))
        if ep % log_every == 0 or ep == 1:
            print(f"epoch {ep:5d}  loss {hist[-1][0]:.3e}  tm {hist[-1][1]:.3e}  "
                  f"consi {hist[-1][2]:.3e}  ({time.time() - t0:.0f}s)", flush=True)
    return params, hist


def run_name(pde, name, rk2=False, lam=C.LAMBDA_CONSI):
    s = f"{pde}_{name}"
    if rk2:
        s += "_rk2"
    if name == "flux_fno" and lam == 0:
        s += "_nocons"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", choices=list(C.PDES), required=True)
    ap.add_argument("--model", choices=list(C.MODELS), default="flux_fno")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--lam", type=float, default=C.LAMBDA_CONSI)
    ap.add_argument("--rk2", action="store_true", help="Flux FNO + RK2 (Algorithme 3)")
    args = ap.parse_args()
    if args.model == "fno_snap" and args.pde != "burgers":
        ap.error("FNO(snap) n'est défini que pour Burgers (section 4.2)")

    print(jax.devices())
    X, Y, batch, dt = load_dataset(args.pde, args.model)
    print(f"{args.model} / {args.pde} : X {X.shape}, Y {Y.shape}, lot {batch}")

    model = build(args.model, use_grid=args.model != "flux_fno")
    key_init, key_train = jax.random.split(jax.random.PRNGKey(C.SEED))
    if args.model == "flux_fno":
        x0 = FF.stencil(jnp.asarray(X[0]))
    else:
        x0 = jnp.asarray(X[:1] if batch > 1 else X[0])[..., None]
    params = model.init(key_init, x0)
    print("paramètres :", sum(p.size for p in jax.tree_util.tree_leaves(params)))

    loss_fn = make_loss(args.pde, args.model, model, dt, args.lam, args.rk2)
    params, hist = fit(loss_fn, params, X, Y, batch, args.epochs, key_train)

    os.makedirs(C.CKPT_DIR, exist_ok=True)
    name = run_name(args.pde, args.model, args.rk2, args.lam)
    with open(os.path.join(C.CKPT_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(dict(params=jax.device_get(params), hist=hist, args=vars(args)), f)
    print("sauvegardé :", name)


if __name__ == "__main__":
    main()
