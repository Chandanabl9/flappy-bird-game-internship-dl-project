"""Training script for Flappy Bird AI using NEAT."""

from __future__ import annotations

import argparse
import os
import sys

from ai_agent import run_neat, replay_best


def main():
    parser = argparse.ArgumentParser(description="Train Flappy Bird AI with NEAT")
    parser.add_argument(
        "--mode", choices=["train", "replay", "play"],
        default="train",
        help="train = evolve AI, replay = watch best genome, play = human mode",
    )
    parser.add_argument(
        "--generations", type=int, default=50,
        help="Number of generations to evolve (default: 50)",
    )
    parser.add_argument(
        "--visual", action="store_true", default=True,
        help="Show Pygame window during training (default: True)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Train without Pygame window (faster)",
    )
    parser.add_argument(
        "--genome-path", type=str, default="outputs/best_genome.pkl",
        help="Path to save/load the best genome",
    )
    parser.add_argument(
        "--neat-config", type=str, default=None,
        help="Path to NEAT config file",
    )

    args = parser.parse_args()

    # Resolve NEAT config path
    config_path = args.neat_config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "neat_config.txt")

    if not os.path.exists(config_path):
        print(f"Error: NEAT config not found at {config_path}")
        sys.exit(1)

    if args.mode == "train":
        visual = not args.headless
        print(f"Starting NEAT evolution ({'visual' if visual else 'headless'} mode)")
        print(f"Generations: {args.generations}")
        print(f"Config: {config_path}")
        print(f"Save to: {args.genome_path}")
        print("-" * 50)

        os.makedirs(os.path.dirname(args.genome_path) or ".", exist_ok=True)
        run_neat(config_path, args.generations, visual=visual,
                 save_path=args.genome_path)

    elif args.mode == "replay":
        if not os.path.exists(args.genome_path):
            print(f"Error: No saved genome at {args.genome_path}")
            print("Train first: python train.py --mode train")
            sys.exit(1)
        print(f"Replaying best genome from {args.genome_path}")
        replay_best(args.genome_path, config_path)

    elif args.mode == "play":
        from game import play_human
        scores_path = os.path.join(os.path.dirname(args.genome_path) or "outputs",
                                   "play_scores.json")
        print("Human play mode — press SPACE to flap, ESC to quit")
        play_human(scores_path=scores_path)


if __name__ == "__main__":
    main()
