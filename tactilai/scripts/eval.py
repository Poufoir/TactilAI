"""
tactilai/scripts/eval.py

Evaluation script — runs trained agents with Pygame renderer.

Usage
─────
  # Watch two trained agents play
  python -m tactilai.scripts.eval --blue checkpoints/blue/checkpoint_002000.pt
                                  --red  checkpoints/red/checkpoint_002000.pt

  # Step-by-step mode
  python -m tactilai.scripts.eval --blue ... --red ... --step

  # Random agents (sanity check, no checkpoint needed)
  python -m tactilai.scripts.eval --random

  # Multiple episodes
  python -m tactilai.scripts.eval --blue ... --red ... --episodes 10
"""

from __future__ import annotations

import argparse

import torch

from tactilai.agents.ppo import PPOAgent
from tactilai.env.gym_wrapper import TactilAIEnv
from tactilai.env.unit import Team
from tactilai.renderer.pygame_renderer import PygameRenderer

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TactilAI — evaluation with Pygame renderer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--blue", type=str, default=None, help="Path to blue agent checkpoint"
    )
    parser.add_argument(
        "--red", type=str, default=None, help="Path to red agent checkpoint"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random agents (no checkpoints needed)",
    )
    parser.add_argument(
        "--step", action="store_true", help="Start in step-by-step mode"
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second in realtime mode"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to play"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    return parser.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────


def get_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Eval loop ─────────────────────────────────────────────────────────────────


def run_episode(
    env: TactilAIEnv,
    agent_blue: PPOAgent,
    agent_red: PPOAgent,
    renderer: PygameRenderer,
    seed: int,
    use_random: bool = False,
) -> str | None:
    """
    Runs one full episode and returns the winner ("BLUE", "RED", or None).
    """
    obs, info = env.reset(seed=seed)

    while True:
        # Render current state
        if not renderer.render(env._grid):
            return None  # user quit

        if env._grid.is_terminal:
            # Render final state once more so user can see the result
            renderer.render(env._grid)
            break

        active_team = env._grid.active_team
        mask = info["action_mask"]

        if use_random:
            action = env.action_space.sample(mask=mask)
        else:
            agent = agent_blue if active_team == Team.BLUE else agent_red
            action, _, _ = agent.select_action(obs, mask)

        obs, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            renderer.render(env._grid)
            break

    return info.get("winner")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    # ── Load agents ───────────────────────────────────────────────────────────
    agent_blue = PPOAgent(device=device)
    agent_red = PPOAgent(device=device)

    if not args.random:
        if args.blue is None or args.red is None:
            print("ERROR: provide --blue and --red checkpoint paths, or use --random")
            return
        agent_blue.load(args.blue)
        agent_red.load(args.red)
        print(f"Loaded blue : {args.blue}")
        print(f"Loaded red  : {args.red}")
    else:
        print("Using random agents.")

    # ── Environment & renderer ────────────────────────────────────────────────
    env = TactilAIEnv(team=Team.BLUE, seed=args.seed)
    obs, info = env.reset(seed=args.seed)

    renderer = PygameRenderer(
        grid=env._grid,
        fps=args.fps,
        step_mode=args.step,
    )

    # ── Episode loop ──────────────────────────────────────────────────────────
    results = {"BLUE": 0, "RED": 0, "DRAW": 0}

    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes} ...", end=" ", flush=True)
        winner = run_episode(
            env,
            agent_blue,
            agent_red,
            renderer,
            seed=args.seed + ep,
            use_random=args.random,
        )
        if winner is None:
            print("(user quit)")
            break

        label = winner if winner else "DRAW"
        results[label] += 1
        print(f"Winner: {label}")

    renderer.close()
    env.close()

    # Summary
    print("\n─── Results ───")
    print(f"BLUE wins : {results['BLUE']}")
    print(f"RED wins  : {results['RED']}")
    print(f"Draws     : {results['DRAW']}")


if __name__ == "__main__":
    main()
