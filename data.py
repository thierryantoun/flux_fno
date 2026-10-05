"""Génération des données (section 4.1) : conditions initiales GRF, solutions
exactes de l'advection, solveur de référence MUSCL-minmod + Godunov + SSP-RK2
pour Burgers, et conditions initiales hors distribution (section 4.2).

Tout est en numpy float64 ; les jeux de données sont stockés en float32.
Usage : python data.py
"""
import os

import numpy as np

import config as C


def grid(n):
    "Centres des mailles sur [0, 1]."
    return (np.arange(n) + 0.5) / n


# ------------------------------------------------------------------ GRF
def sample_grf(rng, n_samples, n=C.N, length=C.GRF_LENGTH):
    """Champ gaussien de covariance k(x,y) = exp(-(d(x,y)/length)^2), où d est
    la distance périodique (le papier ne la précise pas ; on la prend
    périodique pour que la condition initiale soit continue au bord)."""
    x = grid(n)
    d = np.abs(x[:, None] - x[None, :])
    d = np.minimum(d, 1.0 - d)
    K = np.exp(-(d / length) ** 2)
    w, V = np.linalg.eigh(K)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    return rng.standard_normal((n_samples, n)) @ L.T


# ------------------------------------------------------------------ advection
def advection_exact(u0, n_steps, shift=1):
    """Solution exacte de u_t + u_x = 0 avec Δt = Δx : translation d'une maille
    vers la droite par pas. u0 : (..., N) -> (..., n_steps+1, N)."""
    return np.stack([np.roll(u0, k * shift, axis=-1) for k in range(n_steps + 1)], axis=-2)


# ------------------------------------------------------------------ Burgers
def burgers_flux(u):
    return 0.5 * u ** 2


def godunov_flux(uL, uR):
    "Flux de Godunov exact pour Burgers (flux convexe, minimum en 0)."
    return np.maximum(burgers_flux(np.maximum(uL, 0.0)), burgers_flux(np.minimum(uR, 0.0)))


def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def muscl_rhs(u, dx):
    "-(1/Δx)[F_{j+1/2} - F_{j-1/2}] avec reconstruction MUSCL-minmod."
    s = minmod(u - np.roll(u, 1, -1), np.roll(u, -1, -1) - u)
    uL = u + 0.5 * s                                  # état gauche en j+1/2
    uR = np.roll(u - 0.5 * s, -1, -1)                 # état droit en j+1/2
    F = godunov_flux(uL, uR)
    return -(F - np.roll(F, 1, -1)) / dx


def burgers_solve(u0, dt, n_steps, save_every=1):
    """Schéma de Godunov d'ordre 2 (MUSCL-minmod) + SSP-RK2 à pas fixe.
    u0 : (..., N) -> (..., n_steps//save_every + 1, N)."""
    u = np.asarray(u0, dtype=np.float64)
    dx = 1.0 / u.shape[-1]
    out = [u.copy()]
    for k in range(1, n_steps + 1):
        u1 = u + dt * muscl_rhs(u, dx)
        u = 0.5 * u + 0.5 * (u1 + dt * muscl_rhs(u1, dx))
        if k % save_every == 0:
            out.append(u.copy())
    return np.stack(out, axis=-2)


# ------------------------------------------------------------------ OOD
def triangle_pulse(n=C.N, center=0.5, half_width=0.2, height=1.0):
    x = grid(n)
    return height * np.maximum(0.0, 1.0 - np.abs(x - center) / half_width)


def step_function(n=C.N, a=0.25, b=0.75, high=1.0):
    "u0 = high sur [a, b), 0 ailleurs (moyennes de maille exactes)."
    edges = np.arange(n + 1) / n
    return high * np.clip(np.minimum(edges[1:], b) - np.maximum(edges[:-1], a), 0.0, None) * n


def burgers_step_exact(t, n=C.N, a=0.25, b=0.75, n_sub=32):
    """Solution exacte (moyennes de maille) de Burgers pour step_function :
    détente issue de a, choc de vitesse 1/2 issu de b. Valable tant que la
    tête de la détente n'a pas rattrapé le choc (t < 2(b-a)) et que le choc
    n'a pas fait le tour du domaine (repli périodique x -> x mod 1)."""
    def u_line(xs):                       # solution sur la droite réelle
        if t == 0:
            return ((xs >= a) & (xs < b)).astype(float)
        return np.where(xs < a, 0.0,
               np.where(xs < a + t, (xs - a) / t,
               np.where(xs < b + 0.5 * t, 1.0, 0.0)))
    xs = (np.arange(n * n_sub) + 0.5) / (n * n_sub)
    u = u_line(xs) + u_line(xs + 1.0)     # support dans [a, b + t/2) ⊂ [0, 2)
    return u.reshape(n, n_sub).mean(-1)


# ------------------------------------------------------------------ jeux de données
def n_steps(pde, T, n=C.N):
    return int(round(T / (C.PDES[pde]["dt_over_dx"] / n)))


def make_trajectories(pde, u0, T, save_every=1):
    n = u0.shape[-1]
    ns = n_steps(pde, T, n)
    if pde == "advection":
        return advection_exact(u0, ns)[..., ::save_every, :]
    return burgers_solve(u0, C.PDES[pde]["dt_over_dx"] / n, ns, save_every)


def path(name):
    return os.path.join(C.DATA_DIR, name)


def main():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    for pde, P in C.PDES.items():
        u0_train = sample_grf(rng, P["n_train"])
        u0_test = sample_grf(rng, P["n_test"])
        train = make_trajectories(pde, u0_train, P["T_train"])
        # Test : on garde ≈ 1280 instantanés (tous les pas pour l'advection).
        every = max(1, n_steps(pde, P["T_test"]) // 1280)
        test = make_trajectories(pde, u0_test, P["T_test"], save_every=every)
        dt = P["dt_over_dx"] / C.N
        np.savez(path(f"{pde}.npz"), train=train.astype(np.float32), test=test.astype(np.float32),
                 dt=dt, test_dt=dt * every)
        print(f"{pde}: train {train.shape}, test {test.shape}, dt={dt:.3e}, test_dt={dt*every:.3e}")

    # FNO(snap) : couples (u0, u(T_train)) pour Burgers (section 4.2).
    T = C.PDES["burgers"]["T_train"]
    u0 = sample_grf(rng, C.SNAP_N_TRAIN)
    uT = make_trajectories("burgers", u0, T, save_every=n_steps("burgers", T))[:, -1]
    np.savez(path("burgers_snap.npz"), u0=u0.astype(np.float32), uT=uT.astype(np.float32), T=T)
    print(f"burgers_snap: {u0.shape} -> u(t={T})")


if __name__ == "__main__":
    main()
