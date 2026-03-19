"""
tactilai/scripts/train.py

Entry point for TactilAI self-play training.

Usage
─────
  # Basic run
  python -m tactilai.scripts.train

  # Custom hyperparameters
  python -m tactilai.scripts.train --updates 5000 --seed 42

  # Named WandB run
  python -m tactilai.scripts.train --run-name "experiment_v1"

  # Resume from existing checkpoints
  python -m tactilai.scripts.train --resume

  # Monitor live on https://wandb.ai/ton-compte/tactilai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from tactilai.training.selfplay import CHECKPOINT_DIR, TOTAL_UPDATES, SelfPlayTrainer

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TactilAI — self-play RL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=TOTAL_UPDATES,
        help="Total PPO updates to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=CHECKPOINT_DIR,
        help="Directory for checkpoints and pool",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cpu' or 'cuda'. Auto-detected if not set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoints",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom WandB run name",
    )
    return parser.parse_args()


# ── Device detection ──────────────────────────────────────────────────────────


def get_device(override: str | None = None) -> torch.device:
    if override:
        device = torch.device(override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, training on CPU (very slow).")

    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {vram:.1f} GB")
    return device


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    print("\n─── TactilAI Self-Play Training ───")
    print(f"Updates        : {args.updates}")
    print(f"Seed           : {args.seed}")
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Resume         : {args.resume}")
    print(f"WandB run      : {args.run_name or 'auto'}")
    print("───────────────────────────────────\n")

    trainer = SelfPlayTrainer(
        device=device,
        total_updates=args.updates,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        wandb_run_name=args.run_name,
    )

    if args.resume:
        _resume(trainer, args.checkpoint_dir)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted — saving final checkpoints...")
        trainer.pool_blue.save_checkpoint(trainer.agent_blue, trainer.update)
        trainer.pool_red.save_checkpoint(trainer.agent_red, trainer.update)
        trainer.elo.save(Path(args.checkpoint_dir) / "elo.json")
        import wandb

        wandb.finish()
        print("Saved. Exiting.")
        sys.exit(0)

    # Save final state
    trainer.pool_blue.save_checkpoint(trainer.agent_blue, trainer.update)
    trainer.pool_red.save_checkpoint(trainer.agent_red, trainer.update)
    trainer.elo.save(Path(args.checkpoint_dir) / "elo.json")
    print(f"\nFinal ELO:\n{trainer.elo}")


def _resume(trainer: SelfPlayTrainer, checkpoint_dir: str) -> None:
    if not trainer.pool_blue.is_empty:
        trainer.pool_blue.load_latest(trainer.agent_blue)
        print(f"Resumed blue from {trainer.pool_blue._pool[-1].name}")
    if not trainer.pool_red.is_empty:
        trainer.pool_red.load_latest(trainer.agent_red)
        print(f"Resumed red from {trainer.pool_red._pool[-1].name}")

    elo_path = Path(checkpoint_dir) / "elo.json"
    if elo_path.exists():
        trainer.elo = trainer.elo.load(elo_path)
        print(f"Resumed ELO from {elo_path}")


if __name__ == "__main__":
    main()
