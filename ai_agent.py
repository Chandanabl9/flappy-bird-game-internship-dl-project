"""NEAT-based AI agent that learns to play Flappy Bird."""

from __future__ import annotations

import os
import pickle

import neat
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, POPULATION_SIZE,
    MAX_GENERATIONS, FITNESS_THRESHOLD, PIPE_GAP,
)
from game import Bird, FlappyBirdGame


# ═══════════════════════════════════════════════════════════════
# Training state (shared across generations)
# ═══════════════════════════════════════════════════════════════
class TrainingStats:
    """Tracks stats across generations for visualization."""

    def __init__(self):
        self.generation = 0
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
        self.best_score_history: list[int] = []
        self.species_count_history: list[int] = []
        self.current_alive = 0
        self.current_best_score = 0
        self.all_time_best_score = 0
        self.best_genome = None

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness_history,
            "avg_fitness": self.avg_fitness_history,
            "best_score": self.best_score_history,
            "species_count": self.species_count_history,
            "all_time_best_score": self.all_time_best_score,
        }


stats = TrainingStats()


# ═══════════════════════════════════════════════════════════════
# Evaluation function (called by NEAT each generation)
# ═══════════════════════════════════════════════════════════════
def eval_genomes_visual(genomes, config):
    """Evaluate genomes with Pygame visualization."""
    global stats
    stats.generation += 1

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Flappy Bird AI — Gen {stats.generation}")
    clock = pygame.time.Clock()

    # Create birds + neural networks for each genome
    nets = []
    birds = []
    genome_list = []

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        genome_list.append(genome)

    game = FlappyBirdGame(birds)
    running = True

    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                raise SystemExit

        # AI decisions
        for i, bird in enumerate(birds):
            if not bird.alive:
                continue
            inputs = game.get_ai_inputs(bird)
            output = nets[i].activate(inputs)
            if output[0] > 0.5:
                bird.flap()

        game.update()

        # Update fitness
        for i, bird in enumerate(birds):
            if bird.alive:
                # Reward for staying alive
                genome_list[i].fitness += 0.1
                # Extra reward for passing pipes
                genome_list[i].fitness += bird.score * 5.0
                # Reward for staying near center of gap
                pipe = game.get_next_pipe()
                if pipe:
                    gap_center = pipe.gap_y + PIPE_GAP / 2
                    dist_to_center = abs(bird.y - gap_center)
                    genome_list[i].fitness += max(0, (1 - dist_to_center / SCREEN_HEIGHT) * 0.05)

        # Track alive count
        stats.current_alive = sum(1 for b in birds if b.alive)
        stats.current_best_score = game.score

        # Draw
        game.draw(screen, show_ai_info=True)

        # Gen info
        font = pygame.font.SysFont("Arial", 20)
        gen_text = font.render(f"Gen: {stats.generation}", True, (255, 255, 255))
        screen.blit(gen_text, (10, 35))

        pygame.display.flip()
        clock.tick(FPS)

    # End-of-generation stats
    fitnesses = [g.fitness for _, g in genomes]
    stats.best_fitness_history.append(max(fitnesses) if fitnesses else 0)
    stats.avg_fitness_history.append(
        sum(fitnesses) / len(fitnesses) if fitnesses else 0
    )
    stats.best_score_history.append(game.score)
    if game.score > stats.all_time_best_score:
        stats.all_time_best_score = game.score

    pygame.quit()


def eval_genomes_headless(genomes, config):
    """Evaluate genomes without visualization (fast training)."""
    global stats
    stats.generation += 1

    nets = []
    birds = []
    genome_list = []

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        genome_list.append(genome)

    game = FlappyBirdGame(birds)
    max_frames = 10000  # prevent infinite games

    for _ in range(max_frames):
        if game.game_over:
            break

        for i, bird in enumerate(birds):
            if not bird.alive:
                continue
            inputs = game.get_ai_inputs(bird)
            output = nets[i].activate(inputs)
            if output[0] > 0.5:
                bird.flap()

        game.update()

        for i, bird in enumerate(birds):
            if bird.alive:
                genome_list[i].fitness += 0.1
                genome_list[i].fitness += bird.score * 5.0
                pipe = game.get_next_pipe()
                if pipe:
                    gap_center = pipe.gap_y + PIPE_GAP / 2
                    dist_to_center = abs(bird.y - gap_center)
                    genome_list[i].fitness += max(0, (1 - dist_to_center / SCREEN_HEIGHT) * 0.05)

    stats.current_best_score = game.score
    fitnesses = [g.fitness for _, g in genomes]
    stats.best_fitness_history.append(max(fitnesses) if fitnesses else 0)
    stats.avg_fitness_history.append(
        sum(fitnesses) / len(fitnesses) if fitnesses else 0
    )
    stats.best_score_history.append(game.score)
    if game.score > stats.all_time_best_score:
        stats.all_time_best_score = game.score


# ═══════════════════════════════════════════════════════════════
# Run training
# ═══════════════════════════════════════════════════════════════
def run_neat(config_path: str, generations: int = MAX_GENERATIONS,
             visual: bool = True, save_path: str = "best_genome.pkl"):
    """Run NEAT evolution and return the best genome."""
    global stats
    stats = TrainingStats()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stat_reporter = neat.StatisticsReporter()
    population.add_reporter(stat_reporter)

    eval_fn = eval_genomes_visual if visual else eval_genomes_headless
    winner = population.run(eval_fn, generations)

    # Save best genome
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(winner, f)

    stats.best_genome = winner
    print(f"\nBest genome fitness: {winner.fitness:.2f}")
    print(f"Best score achieved: {stats.all_time_best_score}")
    print(f"Saved to: {save_path}")

    return winner, config, stats


# ═══════════════════════════════════════════════════════════════
# Replay best genome
# ═══════════════════════════════════════════════════════════════
def replay_best(genome_path: str, config_path: str):
    """Load a saved genome and watch it play."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird AI — Best Genome Replay")
    clock = pygame.time.Clock()

    bird = Bird()
    game = FlappyBirdGame([bird])
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    bird = Bird()
                    game = FlappyBirdGame([bird])

        if not game.game_over:
            inputs = game.get_ai_inputs(bird)
            output = net.activate(inputs)
            if output[0] > 0.5:
                bird.flap()

        game.update()
        game.draw(screen, show_ai_info=True)

        if game.game_over:
            font = pygame.font.SysFont("Arial", 32)
            text = font.render("AI Done! Press R to restart", True, (255, 255, 0))
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 400))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
