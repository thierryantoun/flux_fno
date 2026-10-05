"""Cœur de la méthode (section 3) : le flux numérique F̂ est remplacé par un FNO.

Schéma conservatif (éq. 2) :
    U_j^{n+1} = U_j^n - Δt/Δx [ G(U_{j-p..j+q}) - G(U_{j-p-1..j+q-1}) ]

G reçoit la fonction entière : les copies décalées (U_{-p}, ..., U_{+q}) sont
concaténées en canaux (Algorithme 1), et la sortie G_j est le flux à
l'interface j+1/2.
"""
import jax.numpy as jnp

import config as C


def physical_flux(pde, u):
    if pde == "advection":
        return C.PDES["advection"]["a"] * u
    return 0.5 * u ** 2


def stencil(u, p=C.STENCIL_P, q=C.STENCIL_Q):
    "(..., N) -> (..., N, p+q+1) ; le canal i contient U_{j+i}, i = -p..q."
    return jnp.stack([jnp.roll(u, -i, axis=-1) for i in range(-p, q + 1)], axis=-1)


def interface_flux(model, params, u):
    "G(U^l) : flux numérique aux interfaces j+1/2, (..., N)."
    return model.apply(params, stencil(u))[..., 0]


def flux_divergence(model, params, u, dx):
    """(1/Δx)[G(U^l) - G(U^r)]. Le FNO sans coordonnée de grille est équivariant
    par translation (périodique), donc G(U^r) = roll(G(U^l), 1) exactement."""
    G = interface_flux(model, params, u)
    return (G - jnp.roll(G, 1, axis=-1)) / dx


def euler_step(model, params, u, dt, dx):
    "Algorithme 2 : un pas de temps du schéma conservatif."
    return u - dt * flux_divergence(model, params, u, dx)


def rk2_step(model, params, u, dt, dx):
    """Algorithmes 3-4 avec la RK2 TVD de Gottlieb-Shu (Heun) :
    U1 = U0 - Δt D(U0) ; U^{n+1} = ½U0 + ½(U1 - Δt D(U1))."""
    u1 = u - dt * flux_divergence(model, params, u, dx)
    return 0.5 * u + 0.5 * (u1 - dt * flux_divergence(model, params, u1, dx))


def consistency_loss(model, params, pde, u, n_in=C.STENCIL_P + C.STENCIL_Q + 1):
    "Résidu de L_consi = ||G(U, ..., U) - F(U)||² (éq. 3), par point."
    G = model.apply(params, jnp.repeat(u[..., None], n_in, axis=-1))[..., 0]
    return (G - physical_flux(pde, u)) ** 2


def loss(model, params, pde, u_n, u_np1, dt, lam=C.LAMBDA_CONSI, rk2=False):
    """L = L_tm + λ L_consi pour un lot de pas consécutifs (u_n, u_{n+1}).
    L_tm = ||U^{n+1} - U^n + Δt/Δx[G(U^l) - G(U^r)]||².

    Comme dans le papier, les normes sont des SOMMES (sur le lot et l'espace),
    pas des moyennes. Ce n'est pas un simple facteur d'échelle : le weight
    decay façon torch.Adam ajoute 1e-3·θ au gradient, et avec une perte
    moyennée (~1e-4) ce terme domine et empêche tout apprentissage.
    Les valeurs renvoyées pour le suivi sont, elles, des moyennes par point."""
    dx = 1.0 / u_n.shape[-1]
    step = rk2_step if rk2 else euler_step
    r_tm = (u_np1 - step(model, params, u_n, dt, dx)) ** 2
    r_c = consistency_loss(model, params, pde, u_n)
    return jnp.sum(r_tm) + lam * jnp.sum(r_c), (jnp.mean(r_tm), jnp.mean(r_c))
