# TactilAI

> A modular reinforcement learning framework for forging competitive agents in turn-based strategy games.

Two independent PPO agents compete in a Fire Emblem-inspired grid environment, trained via adversarial self-play on AMD GPU (ROCm).

## Stack

- **Python 3.12** · **PyTorch 2.6 + ROCm 6.2.4**
- **Gymnasium** — environment interface
- **PPO** — actor-critic with independent replay buffers
- **Pygame** — game renderer
- **TensorBoard** — training metrics

## Quickstart

```bash
# 1. Clone
git clone git@github.com:Poufoir/TactilAI.git && cd TactilAI

# 2. Environnement Conda (inclut PyTorch ROCm)
conda env create -f environment.yml
conda activate fire-emblem-rl

# 3. Dépendances Poetry
poetry install --without dev   # prod
poetry install                 # dev (tests, linter)

# 4. Entraînement
python scripts/train.py

# 5. Évaluation avec rendu Pygame
python scripts/eval.py --render
```

## Structure

```
forge_rl/
├── env/          # Gymnasium environment (grid, units, terrain)
├── agents/       # PPO, networks, replay buffer
├── training/     # self-play loop, curriculum, ELO
├── renderer/     # Pygame renderer
└── utils/        # config, logging, helpers
```

## Roadmap

- [x] Project setup
- [ ] Grid environment (Phase 1)
- [ ] PPO baseline vs heuristic bot (Phase 2)
- [ ] Adversarial self-play (Phase 3)
- [ ] Pygame renderer (Phase 4)
- [ ] Documentation & demo (Phase 5)

## License

MIT