"""
CLI entry point for a single training run.

Example (once all components are implemented):
    python train.py --system A --seed 0 --k 1 --lam 0.0 --episodes 2000
    python train.py --system B --seed 1 --k 5 --lam 1e-3 --episodes 2000

Episode budget can come from config.n_episodes (once pinned) or be
overridden here via --episodes, per the config.py TBD note.
"""

import argparse

from config import Config
from sugarstdp.episode_loop import run_training
from sugarstdp.logger import ExperimentLogger, find_solved_episode
from sugarstdp.system_a import SystemA
from sugarstdp.system_b import SystemB

SYSTEM_BUILDERS = {
    "A": lambda cfg: SystemA(cfg),
    "B": lambda cfg: SystemB(cfg),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train System A or System B on CartPole-v1.")
    parser.add_argument("--system", choices=["A", "B"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=1, help="Reward-delay in environment timesteps.")
    parser.add_argument("--lam", type=float, default=0.0, help="Activity regularization strength.")
    parser.add_argument("--episodes", type=int, default=None, help="Overrides config.n_episodes if set.")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()

    n_episodes = args.episodes if args.episodes is not None else config.n_episodes
    if n_episodes is None:
        raise ValueError(
            "Episode budget not set: pass --episodes or pin config.n_episodes."
        )

    run_name = f"system{args.system}_k{args.k}_lam{args.lam}_seed{args.seed}"
    logger = ExperimentLogger(run_name, config, log_dir=args.log_dir)

    rewards = run_training(
        system_builder=SYSTEM_BUILDERS[args.system],
        config=config,
        seed=args.seed,
        n_episodes=n_episodes,
        k=args.k,
        lam=args.lam,
        logger=logger,
    )
    logger.close()

    if args.system == "A":
        solved_at = find_solved_episode(rewards, config.solve_reward_threshold, config.solve_window)
        if solved_at is not None:
            print(f"System A solved at episode {solved_at} "
                  f"(mean reward >= {config.solve_reward_threshold} over {config.solve_window} episodes).")
        else:
            print(f"System A did not reach the solved gate within {n_episodes} episodes.")


if __name__ == "__main__":
    main()
