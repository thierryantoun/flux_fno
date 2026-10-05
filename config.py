"""Hyperparamètres de « Approximating Numerical Fluxes Using Fourier Neural
Operators for Hyperbolic Conservation Laws » (Kim & Kang, arXiv:2401.01783v3).

Les valeurs suivent les sections 4.1 et les tables 1-2 du papier. Les choix que
le papier ne précise pas sont marqués « (choix) ».
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
FIG_DIR = os.path.join(ROOT, "figures")

SEED = 0

# ---------------------------------------------------------------- maillage
N = 256                    # Δx = 2^-8 sur [0, 1], conditions périodiques

# ---------------------------------------------------------------- GRF
GRF_LENGTH = 0.1           # k(x,y) = exp(-100 (x-y)^2) = exp(-((x-y)/0.1)^2)
GRF_LENGTH_OOD = 0.03      # k(x,y) = exp(-((x-y)/0.03)^2)  (section 4.2)

# ---------------------------------------------------------------- EDP
PDES = {
    "advection": dict(
        # Le papier écrit c = -1 mais dit que la solution « translate vers la
        # droite » : on prend u_t + a u_x = 0 avec a = +1 (flux F(u) = u).
        a=1.0,
        dt_over_dx=1.0,      # Δt = Δx = 2^-8 (translation exacte d'une maille)
        T_train=1.0, T_test=5.0,
        n_train=100, n_test=10,
        batch_size=64,
        eval_times=[0.4, 0.8, 2.5, 5.0],
    ),
    "burgers": dict(
        dt_over_dx=1e-2,     # Δt = 10^-2 · 2^-8
        T_train=0.3, T_test=0.6,
        n_train=10, n_test=10,
        batch_size=100,
        eval_times=[0.15, 0.30, 0.45, 0.60],
    ),
}

# ---------------------------------------------------------------- Flux FNO
# Stencil du flux numérique F̂(U_{j-p}, ..., U_{j+q}) à l'interface j+1/2.
# (choix) p=1, q=2 : le stencil du schéma MUSCL-minmod de référence.
STENCIL_P = 1
STENCIL_Q = 2

MODELS = {
    #                 width  depth  modes
    "flux_fno":  dict(width=64, depth=1, modes=5),
    "fno1d":     dict(width=64, depth=1, modes=5),
    "fno1d_heavy": dict(width=32, depth=3, modes=20),
    "fno2d":     dict(width=64, depth=3, modes=(10, 10)),
    "fno_snap":  dict(width=64, depth=1, modes=5),   # (choix) même archi que 1D FNO
}
PROJ_HIDDEN = 128          # (choix) couche cachée de la projection (FCN à 2 couches)

# ---------------------------------------------------------------- entraînement
LR = 1e-3
WEIGHT_DECAY = 1e-3        # weight decay « à la torch.optim.Adam » (L2 ajouté au gradient)
STEP_SIZE = 50             # StepLR(step_size=50, gamma=0.5), en epochs
GAMMA = 0.5
EPOCHS = 1000
LAMBDA_CONSI = 1e-2

# Baselines (choix pour ce que le papier ne précise pas)
FNO2D_BLOCK = {"advection": 128, "burgers": 100}   # nb de pas de temps par bloc
FNO2D_BATCH = 4
SNAP_N_TRAIN = 1000        # nb de couples (u0, u(T_snap)) pour FNO(snap)
SNAP_BATCH = 50
