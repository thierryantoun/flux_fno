"""FNO 1D / 2D (Li et al. 2021) en Flax.

Architecture commune du papier : relèvement = FCN à 1 couche, projection = FCN
à 2 couches, activation GELU, et dans chaque couche de Fourier la branche
locale est un CNN de noyau 1 (donc sans padding ; le domaine est périodique,
on ne pad pas non plus la FFT).
"""
import jax.numpy as jnp
import flax.linen as nn

import config as C


def _spectral_init(scale):
    def init(key, shape, dtype=jnp.float32):
        return scale * nn.initializers.uniform(1.0)(key, shape, dtype)
    return init


class SpectralConv1d(nn.Module):
    width: int
    modes: int

    @nn.compact
    def __call__(self, v):                                   # (..., N, C)
        c = v.shape[-1]
        init = _spectral_init(1.0 / (c * self.width))
        Rr = self.param("R_real", init, (self.modes, c, self.width))
        Ri = self.param("R_imag", init, (self.modes, c, self.width))
        n = v.shape[-2]
        m = min(self.modes, n // 2 + 1)
        vk = jnp.fft.rfft(v, axis=-2)[..., :m, :]            # (..., m, C)
        out_k = jnp.einsum("...kc,kcd->...kd", vk, (Rr + 1j * Ri)[:m])
        pad = [(0, 0)] * (out_k.ndim - 2) + [(0, n // 2 + 1 - m), (0, 0)]
        return jnp.fft.irfft(jnp.pad(out_k, pad), n=n, axis=-2)


class SpectralConv2d(nn.Module):
    width: int
    modes: tuple

    @nn.compact
    def __call__(self, v):                                   # (..., T, N, C)
        c = v.shape[-1]
        m1, m2 = self.modes
        init = _spectral_init(1.0 / (c * self.width))
        # Deux blocs de poids : fréquences temporelles positives et négatives.
        W = [self.param(f"R{i}_{p}", init, (m1, m2, c, self.width))
             for i in range(2) for p in ("real", "imag")]
        W1, W2 = W[0] + 1j * W[1], W[2] + 1j * W[3]
        T, n = v.shape[-3], v.shape[-2]
        vk = jnp.fft.rfft2(v, axes=(-3, -2))
        out = jnp.zeros(vk.shape[:-1] + (self.width,), dtype=vk.dtype)
        out = out.at[..., :m1, :m2, :].set(jnp.einsum("...abc,abcd->...abd", vk[..., :m1, :m2, :], W1))
        out = out.at[..., -m1:, :m2, :].set(jnp.einsum("...abc,abcd->...abd", vk[..., -m1:, :m2, :], W2))
        return jnp.fft.irfft2(out, s=(T, n), axes=(-3, -2))


class FNO(nn.Module):
    """FNO générique : P (lifting) -> depth × [K + W, GELU] -> Q (projection).

    dim=1 : entrée (..., N, c_in) ; dim=2 : entrée (..., T, N, c_in).
    `use_grid` ajoute la coordonnée spatiale (et temporelle en 2D) en entrée,
    comme dans le FNO standard. Pour le Flux FNO on la désactive : le modèle
    est alors exactement équivariant par translation, ce qui permet de
    calculer G(U^r) comme un simple décalage de G(U^l).
    """
    width: int
    depth: int
    modes: object
    out_dim: int = 1
    dim: int = 1
    use_grid: bool = False

    @nn.compact
    def __call__(self, v):
        if self.use_grid:
            shape = v.shape[:-1]
            grids = jnp.meshgrid(*[(jnp.arange(s) + 0.5) / s for s in shape[-self.dim:]], indexing="ij")
            g = jnp.stack(grids, -1)
            v = jnp.concatenate([v, jnp.broadcast_to(g, shape + (self.dim,))], -1)
        v = nn.Dense(self.width)(v)                          # lifting : FCN 1 couche
        Spec = SpectralConv1d if self.dim == 1 else SpectralConv2d
        for _ in range(self.depth):
            # nn.Dense sur le dernier axe == CNN de noyau 1
            v = nn.gelu(Spec(self.width, self.modes)(v) + nn.Dense(self.width)(v))
        v = nn.gelu(nn.Dense(C.PROJ_HIDDEN)(v))              # projection : FCN 2 couches
        return nn.Dense(self.out_dim)(v)


def build(name, **kw):
    h = C.MODELS[name]
    return FNO(width=h["width"], depth=h["depth"], modes=h["modes"],
               dim=2 if name == "fno2d" else 1, **kw)
