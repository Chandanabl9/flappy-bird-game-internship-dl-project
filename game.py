"""Flappy Bird game engine built with Pygame."""

from __future__ import annotations

import json
import os
import random
import math
import time
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    SKY_BLUE, GREEN, DARK_GREEN, GROUND_COLOR, WHITE, BLACK,
    YELLOW, ORANGE, BIRD_COLOR, BIRD_OUTLINE, PIPE_COLOR, PIPE_BORDER, RED,
    BIRD_X, BIRD_SIZE, GRAVITY, FLAP_STRENGTH, MAX_FALL_SPEED,
    PIPE_WIDTH, PIPE_GAP, PIPE_SPEED, PIPE_SPAWN_INTERVAL,
    MIN_PIPE_HEIGHT, MAX_PIPE_HEIGHT, GROUND_HEIGHT,
)


# ═══════════════════════════════════════════════════════════════
# Bird
# ═══════════════════════════════════════════════════════════════
class Bird:
    def __init__(self, y: float | None = None):
        self.x = BIRD_X
        self.y = y if y is not None else SCREEN_HEIGHT // 2
        self.velocity = 0.0
        self.alive = True
        self.score = 0
        self.distance = 0  # frames survived
        self.flap_angle = 0.0

    def flap(self):
        if self.alive:
            self.velocity = FLAP_STRENGTH

    def update(self):
        if not self.alive:
            return
        self.velocity += GRAVITY
        self.velocity = min(self.velocity, MAX_FALL_SPEED)
        self.y += self.velocity
        self.distance += 1

        # Flap animation angle
        self.flap_angle = max(-30, min(60, self.velocity * 4))

        # Floor / ceiling
        if self.y < 0:
            self.y = 0
            self.velocity = 0
        if self.y > SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_SIZE:
            self.y = SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_SIZE
            self.alive = False

    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, int(self.y), BIRD_SIZE, BIRD_SIZE)

    def draw(self, surface: pygame.Surface):
        # Body
        cx = self.x + BIRD_SIZE // 2
        cy = int(self.y) + BIRD_SIZE // 2
        pygame.draw.circle(surface, BIRD_COLOR, (cx, cy), BIRD_SIZE // 2)
        pygame.draw.circle(surface, BIRD_OUTLINE, (cx, cy), BIRD_SIZE // 2, 2)

        # Eye
        eye_x = cx + 6
        eye_y = cy - 4
        pygame.draw.circle(surface, WHITE, (eye_x, eye_y), 5)
        pygame.draw.circle(surface, BLACK, (eye_x + 1, eye_y), 2)

        # Beak
        beak_points = [(cx + BIRD_SIZE // 2, cy),
                       (cx + BIRD_SIZE // 2 + 8, cy + 2),
                       (cx + BIRD_SIZE // 2, cy + 5)]
        pygame.draw.polygon(surface, ORANGE, beak_points)

        # Wing (animated)
        wing_offset = int(math.sin(self.distance * 0.3) * 4)
        wing_points = [(cx - 4, cy + 2),
                       (cx - 14, cy + wing_offset),
                       (cx - 4, cy + 8)]
        pygame.draw.polygon(surface, YELLOW, wing_points)
        pygame.draw.polygon(surface, BIRD_OUTLINE, wing_points, 1)


# ═══════════════════════════════════════════════════════════════
# Pipe
# ═══════════════════════════════════════════════════════════════
class Pipe:
    def __init__(self, x: float | None = None, gap_y: int | None = None):
        self.x = x if x is not None else SCREEN_WIDTH + 50
        if gap_y is not None:
            self.gap_y = gap_y
        else:
            self.gap_y = random.randint(MIN_PIPE_HEIGHT, MAX_PIPE_HEIGHT)
        self.scored = False

    @property
    def top_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), 0, PIPE_WIDTH, self.gap_y)

    @property
    def bottom_rect(self) -> pygame.Rect:
        top = self.gap_y + PIPE_GAP
        return pygame.Rect(int(self.x), top, PIPE_WIDTH,
                           SCREEN_HEIGHT - top - GROUND_HEIGHT)

    def update(self):
        self.x -= PIPE_SPEED

    def off_screen(self) -> bool:
        return self.x + PIPE_WIDTH < 0

    def draw(self, surface: pygame.Surface):
        # Top pipe
        tr = self.top_rect
        pygame.draw.rect(surface, PIPE_COLOR, tr)
        pygame.draw.rect(surface, PIPE_BORDER, tr, 3)
        # Lip
        lip = pygame.Rect(tr.x - 4, tr.bottom - 24, PIPE_WIDTH + 8, 24)
        pygame.draw.rect(surface, PIPE_COLOR, lip)
        pygame.draw.rect(surface, PIPE_BORDER, lip, 3)

        # Bottom pipe
        br = self.bottom_rect
        pygame.draw.rect(surface, PIPE_COLOR, br)
        pygame.draw.rect(surface, PIPE_BORDER, br, 3)
        # Lip
        lip2 = pygame.Rect(br.x - 4, br.top, PIPE_WIDTH + 8, 24)
        pygame.draw.rect(surface, PIPE_COLOR, lip2)
        pygame.draw.rect(surface, PIPE_BORDER, lip2, 3)


# ═══════════════════════════════════════════════════════════════
# Game
# ═══════════════════════════════════════════════════════════════
class FlappyBirdGame:
    """Core game logic — usable for both human play and AI training."""

    def __init__(self, birds: list[Bird] | None = None):
        self.birds = birds if birds else [Bird()]
        self.pipes: list[Pipe] = []
        self.frame_count = 0
        self.score = 0
        self.game_over = False
        self.ground_scroll = 0

    # ── helpers ──────────────────────────────────────────────
    def get_next_pipe(self) -> Pipe | None:
        """Return the nearest pipe in front of the bird."""
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > BIRD_X:
                return pipe
        return None

    def get_ai_inputs(self, bird: Bird) -> list[float]:
        """Return normalised inputs for the neural network."""
        pipe = self.get_next_pipe()
        if pipe is None:
            return [bird.y / SCREEN_HEIGHT,
                    bird.velocity / MAX_FALL_SPEED,
                    1.0, 0.5, 0.5]
        return [
            bird.y / SCREEN_HEIGHT,
            bird.velocity / MAX_FALL_SPEED,
            (pipe.x - bird.x) / SCREEN_WIDTH,
            pipe.gap_y / SCREEN_HEIGHT,
            (pipe.gap_y + PIPE_GAP) / SCREEN_HEIGHT,
        ]

    # ── update ───────────────────────────────────────────────
    def update(self):
        if self.game_over:
            return

        self.frame_count += 1
        self.ground_scroll = (self.ground_scroll + PIPE_SPEED) % 40

        # Spawn pipes
        if self.frame_count % PIPE_SPAWN_INTERVAL == 0:
            self.pipes.append(Pipe())

        # Update pipes
        for pipe in self.pipes:
            pipe.update()
        self.pipes = [p for p in self.pipes if not p.off_screen()]

        # Update birds
        alive_count = 0
        for bird in self.birds:
            bird.update()
            if bird.alive:
                alive_count += 1
                # Collision detection
                bird_rect = bird.get_rect()
                for pipe in self.pipes:
                    if bird_rect.colliderect(pipe.top_rect) or \
                       bird_rect.colliderect(pipe.bottom_rect):
                        bird.alive = False
                        break

                # Scoring
                for pipe in self.pipes:
                    if not pipe.scored and pipe.x + PIPE_WIDTH < bird.x:
                        pipe.scored = True
                        bird.score += 1

        # Highest score among all birds
        self.score = max((b.score for b in self.birds), default=0)

        if alive_count == 0:
            self.game_over = True

    # ── draw ─────────────────────────────────────────────────
    def draw(self, surface: pygame.Surface, show_ai_info: bool = False):
        # Sky
        surface.fill(SKY_BLUE)

        # Clouds (decorative)
        self._draw_clouds(surface)

        # Pipes
        for pipe in self.pipes:
            pipe.draw(surface)

        # Ground
        ground_y = SCREEN_HEIGHT - GROUND_HEIGHT
        pygame.draw.rect(surface, GROUND_COLOR,
                         (0, ground_y, SCREEN_WIDTH, GROUND_HEIGHT))
        # Ground texture
        for i in range(0, SCREEN_WIDTH + 40, 40):
            x = i - int(self.ground_scroll)
            pygame.draw.line(surface, (180, 140, 100),
                             (x, ground_y), (x + 20, ground_y + 15), 2)

        # Birds
        alive_count = 0
        for bird in self.birds:
            if bird.alive:
                bird.draw(surface)
                alive_count += 1

        # Score
        font = pygame.font.SysFont("Arial", 48, bold=True)
        score_text = font.render(str(self.score), True, WHITE)
        shadow = font.render(str(self.score), True, BLACK)
        surface.blit(shadow, (SCREEN_WIDTH // 2 - score_text.get_width() // 2 + 2, 52))
        surface.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 50))

        # AI info overlay
        if show_ai_info:
            info_font = pygame.font.SysFont("Arial", 20)
            alive_text = info_font.render(f"Alive: {alive_count}/{len(self.birds)}",
                                          True, WHITE)
            surface.blit(alive_text, (10, 10))

    def _draw_clouds(self, surface: pygame.Surface):
        cloud_positions = [(100, 100), (300, 60), (500, 120), (150, 200), (450, 180)]
        for cx, cy in cloud_positions:
            shifted_x = (cx - self.ground_scroll * 0.2) % (SCREEN_WIDTH + 80) - 40
            for dx, dy, r in [(-15, 0, 20), (0, -8, 25), (15, 0, 20), (0, 5, 18)]:
                pygame.draw.circle(surface, WHITE,
                                   (int(shifted_x + dx), cy + dy), r)

    # ── game over screen ─────────────────────────────────────
    def draw_game_over(self, surface: pygame.Surface):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        surface.blit(overlay, (0, 0))

        font_big = pygame.font.SysFont("Arial", 64, bold=True)
        font_med = pygame.font.SysFont("Arial", 32)

        go_text = font_big.render("GAME OVER", True, RED)
        surface.blit(go_text, (SCREEN_WIDTH // 2 - go_text.get_width() // 2, 280))

        score_text = font_med.render(f"Score: {self.score}", True, WHITE)
        surface.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 370))

        restart_text = font_med.render("Press SPACE to restart", True, YELLOW)
        surface.blit(restart_text,
                     (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 430))


# ═══════════════════════════════════════════════════════════════
# Human play mode (standalone)
# ═══════════════════════════════════════════════════════════════
def play_human(scores_path: str | None = None):
    """Run Flappy Bird in human-play mode. Saves session results to scores_path."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird — Human Mode")
    clock = pygame.time.Clock()

    game = FlappyBirdGame()
    running = True
    session_scores: list[int] = []
    game_start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if game.game_over:
                        session_scores.append(game.score)
                        game = FlappyBirdGame()
                    else:
                        game.birds[0].flap()
                if event.key == pygame.K_ESCAPE:
                    running = False

        game.update()
        game.draw(screen)
        if game.game_over:
            game.draw_game_over(screen)

        pygame.display.flip()
        clock.tick(FPS)

    # Capture final game score if game was in progress or ended
    if game.score > 0 or game.game_over:
        session_scores.append(game.score)

    pygame.quit()

    # Save results
    if scores_path and session_scores:
        total_time = round(time.time() - game_start_time, 1)
        results = {
            "scores": session_scores,
            "top_score": max(session_scores),
            "games_played": len(session_scores),
            "average_score": round(sum(session_scores) / len(session_scores), 1),
            "total_time_seconds": total_time,
        }
        os.makedirs(os.path.dirname(scores_path) or ".", exist_ok=True)
        with open(scores_path, "w") as f:
            json.dump(results, f, indent=2)

    return session_scores


if __name__ == "__main__":
    play_human()
