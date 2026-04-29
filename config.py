"""Game and AI configuration for Flappy Bird."""

from __future__ import annotations

# ── Screen ───────────────────────────────────────────────────
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
FPS = 60

# ── Colors ───────────────────────────────────────────────────
SKY_BLUE = (135, 206, 235)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
GROUND_COLOR = (222, 184, 135)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
YELLOW = (255, 215, 0)
ORANGE = (255, 165, 0)
BIRD_COLOR = (255, 200, 50)
BIRD_OUTLINE = (200, 150, 0)
PIPE_COLOR = GREEN
PIPE_BORDER = DARK_GREEN

# ── Bird ─────────────────────────────────────────────────────
BIRD_X = 80
BIRD_SIZE = 30
GRAVITY = 0.5
FLAP_STRENGTH = -9
MAX_FALL_SPEED = 12

# ── Pipes ────────────────────────────────────────────────────
PIPE_WIDTH = 70
PIPE_GAP = 180
PIPE_SPEED = 4
PIPE_SPAWN_INTERVAL = 90  # frames between pipe spawns
MIN_PIPE_HEIGHT = 80
MAX_PIPE_HEIGHT = SCREEN_HEIGHT - PIPE_GAP - 120  # leave room for ground

# ── Ground ───────────────────────────────────────────────────
GROUND_HEIGHT = 80

# ── AI / NEAT ────────────────────────────────────────────────
POPULATION_SIZE = 50
MAX_GENERATIONS = 50
FITNESS_THRESHOLD = 1000  # score to consider "solved"
