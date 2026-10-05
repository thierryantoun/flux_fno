# Flux FNO — implémentation de arXiv:2401.01783v3

Kim & Kang, *Approximating Numerical Fluxes Using Fourier Neural Operators for
Hyperbolic Conservation Laws*. Le flux numérique F̂ d'un schéma volumes finis
conservatif est remplacé par un FNO G :

    U_j^{n+1} = U_j^n - Δt/Δx [ G(U_{j-p..j+q}) - G(U_{j-p-1..j+q-1}) ]

entraîné avec  L = L_tm + λ L_consi,  L_consi = ||G(U,…,U) - F(U)||²,  λ = 0.01.

## Fichiers

| fichier | rôle | papier |
|---|---|---|
| `config.py` | hyperparamètres | tables 1-2, §4.1 |
| `data.py` | GRF, advection exacte, Burgers MUSCL-minmod + Godunov + SSP-RK2, CI hors distribution | §4.1, §4.2 |
| `models.py` | FNO 1D/2D (lifting 1 couche, projection 2 couches, GELU, CNN noyau 1) | §2.3, §4.1 |
| `flux_fno.py` | stencil, schéma conservatif, RK2, pertes L_tm / L_consi | §3, alg. 1-4 |
| `train.py` | entraînement Flux FNO + baselines | alg. 1, 3 |
| `evaluate.py` | tables 3-6, figures (OOD, résolutions 128/512) | §4 |

## Utilisation (environnement `test-env` : jax, flax, optax)

```bash
python data.py                                         # ~2 min

# §4.1 : Flux FNO et baselines
python train.py --pde advection --model flux_fno
python train.py --pde burgers   --model flux_fno
python train.py --pde burgers   --model fno1d          # idem fno1d_heavy, fno2d
python train.py --pde burgers   --model fno_snap       # §4.2, FNO(snap)
# §4.3 : RK2      §4.4 : ablation sans perte de consistance
python train.py --pde burgers   --model flux_fno --rk2
python train.py --pde burgers   --model flux_fno --lam 0

python evaluate.py --pde burgers                       # tableaux + figures/
python evaluate.py --pde advection
```

`--epochs N` raccourcit l'entraînement (le papier utilise 1000 epochs). Sur CPU,
une epoch de Burgers prend environ 1 à 2 min. Pour les 1000 epochs, il faut un
GPU (`pip install -U "jax[cuda12]"`).

## Choix d'implémentation (non précisés ou ambigus dans le papier)

- **Stencil** : p = 1, q = 2 (celui du schéma MUSCL de référence). Les canaux
  d'entrée de G sont les copies décalées U_{j-1}, U_j, U_{j+1}, U_{j+2}.
- **G(U^r)** : le FNO du Flux FNO ne reçoit pas la coordonnée de grille. Il est
  donc exactement équivariant par translation périodique, et on calcule
  G(U^r) = roll(G(U^l), 1) au lieu d'un second passage dans le réseau.
- **Advection** : le papier écrit c = -1 mais décrit une translation vers la
  droite. On prend u_t + u_x = 0 (F(u) = u) : avec Δt = Δx, la solution exacte
  avance d'une maille par pas.
- **GRF** : covariance exp(-((x-y)/0.1)²), avec une distance périodique pour que
  la CI soit continue au bord.
- **L_tm** : moyenne au lieu de somme (simple facteur d'échelle). Pour Burgers
  (Δt/Δx = 0.01), L_tm est numériquement bien plus petite que λ·L_consi, comme
  dans la formulation du papier.
- **RK2** : RK2 TVD de Gottlieb-Shu (α, β = ½).
- **Weight decay** : sous la forme de `torch.optim.Adam` (L2 ajouté au gradient),
  et non AdamW. StepLR est compté en epochs.
- **2D FNO** : bloc de Tb pas de temps → bloc suivant (Tb = 128 pour
  l'advection, 100 pour Burgers). En inférence, le premier bloc est donné par
  la référence.
- **FNO(snap)** : 1000 couples (u0, u(0.3)) ; appliqué deux fois pour t = 0.6.
- **OOD** : triangle de demi-largeur 0.2 ; marche 1 sur [0.25, 0.75) avec sa
  solution exacte (détente + choc, repli périodique).

## Résultats de référence du papier (L² relative, L∞)

| | advection t=5.0 | Burgers t=0.6 |
|---|---|---|
| Flux FNO | (1.04e-2, 4.53e-2) | (0.052, 0.13) |
| Flux FNO + RK2 | — | (0.041, 0.10) |
| 2D FNO | (1.94e-2, 4.67e-2) | (2.21, 1.26) |
| 1D FNO (heavy) | (0.625, 1.19) | (11.20, 7.32) |
