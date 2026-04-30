"""Streamlit dashboard for Flappy Bird AI project."""

from __future__ import annotations

import os
import subprocess
import json

import streamlit as st
import numpy as np

# Resolve project directory reliably (PROJECT_DIR can be empty in Streamlit)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Detect if running on Streamlit Cloud (no display, no pygame)
IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") or not os.environ.get("DISPLAY", os.name == "nt")

# Try importing pickle and neat for results page
try:
    import pickle
    import neat
    NEAT_AVAILABLE = True
except ImportError:
    NEAT_AVAILABLE = False

st.set_page_config(page_title="Flappy Bird AI", page_icon="🐦", layout="wide")

page = st.sidebar.radio("Navigate", ["Home", "Train AI", "Play Game", "Results"])

# ═══════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════
if page == "Home":
    st.title("🐦 Flappy Bird AI")
    st.markdown("##### Teaching a Neural Network to Play Flappy Bird using NEAT Evolution")

    st.divider()

    st.header("Project Introduction")
    st.markdown(
        """
        **Flappy Bird** is a classic side-scrolling game where a bird must fly
        through gaps between pipes. One tap makes the bird flap upward; gravity
        pulls it down. Hit a pipe or the ground and it's game over.

        This project uses **NEAT (NeuroEvolution of Augmenting Topologies)** —
        a genetic algorithm that evolves neural networks — to create an AI
        that learns to play Flappy Bird **from scratch** without any human
        training data.
        """
    )

    st.divider()

    st.header("How NEAT Works")
    st.markdown(
        """
        | Step | What Happens |
        |---|---|
        | **1. Population** | Create 50 birds, each with a random neural network |
        | **2. Play** | All 50 birds play Flappy Bird simultaneously |
        | **3. Fitness** | Each bird gets a score based on how far it survived |
        | **4. Selection** | Top-performing birds are selected as "parents" |
        | **5. Crossover** | Parent neural networks are combined to create offspring |
        | **6. Mutation** | Random changes to weights, connections, and nodes |
        | **7. Repeat** | New generation plays → better birds emerge |

        After **5-10 generations** (~2 minutes), the AI typically learns to
        play perfectly — navigating pipes indefinitely!
        """
    )

    st.divider()

    st.header("Neural Network Architecture")
    st.code(
        """
        INPUTS (5 neurons)              HIDDEN             OUTPUT (1 neuron)
        ┌──────────────────┐                               ┌──────────────┐
        │ Bird Y position  │──┐                        ┌──▶│              │
        │ Bird velocity    │──┤    ┌─────────────┐     │   │  Flap?       │
        │ Distance to pipe │──┼───▶│ Evolved via  │─────┘   │  > 0.5 = YES│
        │ Pipe top Y       │──┤    │ NEAT         │         │  < 0.5 = NO │
        │ Pipe bottom Y    │──┘    └─────────────┘         └──────────────┘
        └──────────────────┘

        • Starts with NO hidden nodes (direct input→output)
        • NEAT adds nodes and connections through mutation
        • Each generation, networks grow more capable
        """,
        language="text",
    )

    st.divider()

    st.header("Implementation Details")

    st.markdown("#### Game Engine (Pygame)")
    st.code(
        """# Bird physics
bird.velocity += GRAVITY       # 0.5 px/frame²
bird.y += bird.velocity
if flap:
    bird.velocity = -9         # instant upward boost

# Pipe collision
bird_rect.colliderect(pipe.top_rect)
bird_rect.colliderect(pipe.bottom_rect)""",
        language="python",
    )

    st.markdown("#### Fitness Function")
    st.code(
        """# How we score each bird:
fitness += 0.1                          # per frame alive
fitness += bird.score * 5.0             # per pipe passed
fitness += (1 - dist_to_gap_center) * 0.05  # staying centered""",
        language="python",
    )

    st.markdown("#### NEAT Evolution")
    st.code(
        """import neat

config = neat.Config(...)
population = neat.Population(config)

# Each generation: evaluate all genomes
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Run the game, set genome.fitness based on performance

winner = population.run(eval_genomes, generations=50)""",
        language="python",
    )

    st.divider()

    st.header("System Architecture")
    st.code(
        """
┌─────────────────────────────────────────────────────────────┐
│                    Flappy Bird AI                            │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │  Game Engine     │    │  NEAT Evolution               │    │
│  │  (Pygame)        │◄──▶│                               │    │
│  │                  │    │  Population of 50 genomes     │    │
│  │  • Bird physics  │    │  ↓                            │    │
│  │  • Pipe spawning │    │  Evaluate (play game)         │    │
│  │  • Collision     │    │  ↓                            │    │
│  │  • Rendering     │    │  Select best performers       │    │
│  │                  │    │  ↓                            │    │
│  └─────────────────┘    │  Crossover + Mutate            │    │
│                          │  ↓                            │    │
│  ┌─────────────────┐    │  New generation                │    │
│  │  Streamlit       │    │  ↓ Repeat                     │    │
│  │  Dashboard       │    └──────────────────────────────┘    │
│  │                  │                                        │
│  │  • Training      │    ┌──────────────────────────────┐    │
│  │    controls      │    │  Neural Network               │    │
│  │  • Live stats    │    │  5 inputs → evolved → 1 out   │    │
│  │  • Fitness       │    │  "Should I flap?"             │    │
│  │    graphs        │    └──────────────────────────────┘    │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘""",
        language="text",
    )

    st.divider()
    st.markdown("👈 Use the sidebar to **Train AI**, **Play** the game, or view **Results**.")

# ═══════════════════════════════════════════════════════════════
# TRAIN AI
# ═══════════════════════════════════════════════════════════════
elif page == "Train AI":
    st.title("🧠 Train the AI")
    st.markdown("Start NEAT evolution to teach the AI to play Flappy Bird.")

    st.info(
        "⚠️ **Training requires a local machine** with pygame and a display.\n\n"
        "This feature does not work on Streamlit Cloud. To train locally, run:\n"
        "```bash\n"
        "pip install pygame neat-python\n"
        "python3 train.py --mode train --generations 30\n"
        "```"
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        generations = st.slider("Number of generations", 5, 100, 30)
        mode = st.radio("Training mode", ["Visual (watch AI play)", "Headless (fast, no window)"])
    with col2:
        st.markdown(
            """
            **Visual mode**: Opens a Pygame window where you can watch
            all 50 birds playing simultaneously. Slower but fun to watch.

            **Headless mode**: No window, trains much faster.
            Results are saved for viewing in the Results tab.
            """
        )

    output_path = os.path.join(PROJECT_DIR, "outputs", "best_genome.pkl")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Check if we can actually run training
        try:
            import pygame
            import neat
        except ImportError:
            st.error(
                "**pygame** and **neat-python** are not installed in this environment.\n\n"
                "To train the AI, run locally:\n"
                "```bash\n"
                "pip install pygame neat-python\n"
                "python3 train.py --mode train --generations " + str(generations) + "\n"
                "```"
            )
            st.stop()

        headless_flag = "--headless" if "Headless" in mode else ""
        cmd = (
            f"cd {PROJECT_DIR} && "
            f"python3 train.py --mode train "
            f"--generations {generations} "
            f"--genome-path {output_path} "
            f"{headless_flag}"
        )

        st.info(f"Running: `{cmd}`")
        st.warning(
            "Training runs in a **separate Pygame window**. "
            "Come back here after it finishes to see results."
        )

        with st.spinner("Training in progress... (check the Pygame window)"):
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=600
                )
                if result.returncode == 0:
                    st.success("Training complete! Go to **Results** tab to see stats.")
                    st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                else:
                    st.error("Training failed:")
                    st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            except subprocess.TimeoutExpired:
                st.warning("Training timed out (10 min limit). Check the Pygame window.")

    st.divider()
    st.subheader("Training Parameters")
    st.markdown(
        f"""
        | Parameter | Value |
        |---|---|
        | Population size | 50 birds per generation |
        | Generations | {generations} |
        | Network inputs | 5 (bird Y, velocity, pipe dist, gap top, gap bottom) |
        | Network output | 1 (flap probability) |
        | Fitness | frames survived × 0.1 + pipes passed × 5.0 |
        | Selection | Top 20% survive, rest are replaced |
        """
    )

# ═══════════════════════════════════════════════════════════════
# PLAY GAME (Browser-based HTML5 Canvas game)
# ═══════════════════════════════════════════════════════════════
elif page == "Play Game":
    st.title("🎮 Play Flappy Bird")
    st.markdown("**Click the game area** then press **SPACE** or **click** to flap! Works in the browser — no install needed.")

    import streamlit.components.v1 as components

    game_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * { margin: 0; padding: 0; }
        canvas { display: block; margin: 0 auto; background: #87CEEB; border-radius: 8px; cursor: pointer; }
        #game-container { text-align: center; font-family: Arial, sans-serif; }
        #score-display { font-size: 18px; margin-top: 10px; color: #333; }
    </style>
    </head>
    <body>
    <div id="game-container">
        <canvas id="gameCanvas" width="400" height="600"></canvas>
        <div id="score-display">Click the game to start | SPACE or Click to flap</div>
    </div>
    <script>
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    // Game state
    let bird = { x: 80, y: H/2, vy: 0, size: 20 };
    let pipes = [];
    let score = 0;
    let bestScore = 0;
    let gameState = 'waiting'; // waiting, playing, dead
    let frame = 0;

    // Physics
    const GRAVITY = 0.4;
    const FLAP = -7;
    const PIPE_SPEED = 3;
    const PIPE_WIDTH = 55;
    const PIPE_GAP = 150;
    const GROUND_H = 60;
    const SPAWN_INTERVAL = 90;

    function reset() {
        bird = { x: 80, y: H/2, vy: 0, size: 20 };
        pipes = [];
        score = 0;
        frame = 0;
        gameState = 'playing';
    }

    function flap() {
        if (gameState === 'waiting') { reset(); return; }
        if (gameState === 'dead') { reset(); return; }
        bird.vy = FLAP;
    }

    canvas.addEventListener('click', flap);
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') { e.preventDefault(); flap(); }
    });

    function spawnPipe() {
        let gapY = 100 + Math.random() * (H - GROUND_H - PIPE_GAP - 200);
        pipes.push({ x: W + 10, gapY: gapY, scored: false });
    }

    function update() {
        if (gameState !== 'playing') return;
        frame++;

        // Bird physics
        bird.vy += GRAVITY;
        bird.vy = Math.min(bird.vy, 10);
        bird.y += bird.vy;

        // Ground/ceiling
        if (bird.y < 0) { bird.y = 0; bird.vy = 0; }
        if (bird.y + bird.size > H - GROUND_H) { gameState = 'dead'; bestScore = Math.max(bestScore, score); }

        // Spawn pipes
        if (frame % SPAWN_INTERVAL === 0) spawnPipe();

        // Move pipes
        for (let p of pipes) {
            p.x -= PIPE_SPEED;
            // Scoring
            if (!p.scored && p.x + PIPE_WIDTH < bird.x) {
                p.scored = true;
                score++;
            }
            // Collision
            if (bird.x + bird.size > p.x && bird.x < p.x + PIPE_WIDTH) {
                if (bird.y < p.gapY || bird.y + bird.size > p.gapY + PIPE_GAP) {
                    gameState = 'dead';
                    bestScore = Math.max(bestScore, score);
                }
            }
        }
        pipes = pipes.filter(p => p.x + PIPE_WIDTH > -10);
    }

    function draw() {
        // Sky gradient
        let grad = ctx.createLinearGradient(0, 0, 0, H);
        grad.addColorStop(0, '#87CEEB');
        grad.addColorStop(1, '#E0F7FA');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H);

        // Clouds
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        [[60,80],[200,50],[320,90],[150,140]].forEach(([cx,cy]) => {
            ctx.beginPath(); ctx.arc(cx,cy,25,0,Math.PI*2); ctx.fill();
            ctx.beginPath(); ctx.arc(cx+20,cy-5,20,0,Math.PI*2); ctx.fill();
            ctx.beginPath(); ctx.arc(cx-15,cy+5,18,0,Math.PI*2); ctx.fill();
        });

        // Pipes
        for (let p of pipes) {
            // Top pipe
            ctx.fillStyle = '#2E8B57';
            ctx.fillRect(p.x, 0, PIPE_WIDTH, p.gapY);
            ctx.fillStyle = '#228B22';
            ctx.fillRect(p.x - 3, p.gapY - 20, PIPE_WIDTH + 6, 20);

            // Bottom pipe
            ctx.fillStyle = '#2E8B57';
            ctx.fillRect(p.x, p.gapY + PIPE_GAP, PIPE_WIDTH, H - p.gapY - PIPE_GAP - GROUND_H);
            ctx.fillStyle = '#228B22';
            ctx.fillRect(p.x - 3, p.gapY + PIPE_GAP, PIPE_WIDTH + 6, 20);

            // Pipe borders
            ctx.strokeStyle = '#1B5E20';
            ctx.lineWidth = 2;
            ctx.strokeRect(p.x, 0, PIPE_WIDTH, p.gapY);
            ctx.strokeRect(p.x, p.gapY + PIPE_GAP, PIPE_WIDTH, H - p.gapY - PIPE_GAP - GROUND_H);
        }

        // Ground
        ctx.fillStyle = '#DEB887';
        ctx.fillRect(0, H - GROUND_H, W, GROUND_H);
        ctx.fillStyle = '#8B7355';
        for (let i = 0; i < W; i += 30) {
            ctx.fillRect(i, H - GROUND_H, 2, 10);
        }
        ctx.strokeStyle = '#654321';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0, H-GROUND_H); ctx.lineTo(W, H-GROUND_H); ctx.stroke();

        // Bird
        let bx = bird.x + bird.size/2, by = bird.y + bird.size/2;
        // Body
        ctx.fillStyle = '#FFD700';
        ctx.beginPath(); ctx.arc(bx, by, bird.size/2 + 2, 0, Math.PI*2); ctx.fill();
        ctx.strokeStyle = '#DAA520';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(bx, by, bird.size/2 + 2, 0, Math.PI*2); ctx.stroke();
        // Eye
        ctx.fillStyle = 'white';
        ctx.beginPath(); ctx.arc(bx+5, by-4, 5, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = 'black';
        ctx.beginPath(); ctx.arc(bx+6, by-4, 2.5, 0, Math.PI*2); ctx.fill();
        // Beak
        ctx.fillStyle = '#FF6600';
        ctx.beginPath(); ctx.moveTo(bx+12,by); ctx.lineTo(bx+20,by+2); ctx.lineTo(bx+12,by+5); ctx.fill();
        // Wing
        let wingY = Math.sin(frame * 0.2) * 3;
        ctx.fillStyle = '#FFA500';
        ctx.beginPath(); ctx.moveTo(bx-5,by+2); ctx.lineTo(bx-15,by+wingY); ctx.lineTo(bx-5,by+8); ctx.fill();

        // Score
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.font = 'bold 36px Arial';
        ctx.textAlign = 'center';
        ctx.strokeText(score, W/2, 50);
        ctx.fillText(score, W/2, 50);

        // Game states
        if (gameState === 'waiting') {
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(0, 0, W, H);
            ctx.fillStyle = 'white';
            ctx.font = 'bold 28px Arial';
            ctx.fillText('🐦 FLAPPY BIRD', W/2, H/2 - 40);
            ctx.font = '20px Arial';
            ctx.fillText('Click or press SPACE to start', W/2, H/2 + 10);
            ctx.font = '16px Arial';
            ctx.fillText('Best: ' + bestScore, W/2, H/2 + 50);
        }

        if (gameState === 'dead') {
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(0, 0, W, H);
            ctx.fillStyle = '#FF4444';
            ctx.font = 'bold 36px Arial';
            ctx.fillText('GAME OVER', W/2, H/2 - 50);
            ctx.fillStyle = 'white';
            ctx.font = '24px Arial';
            ctx.fillText('Score: ' + score, W/2, H/2);
            ctx.fillText('Best: ' + bestScore, W/2, H/2 + 35);
            ctx.font = '18px Arial';
            ctx.fillStyle = '#FFD700';
            ctx.fillText('Click or SPACE to restart', W/2, H/2 + 80);
        }
    }

    function gameLoop() {
        update();
        draw();
        document.getElementById('score-display').textContent =
            'Score: ' + score + ' | Best: ' + bestScore + ' | SPACE or Click to flap';
        requestAnimationFrame(gameLoop);
    }

    gameLoop();
    </script>
    </body>
    </html>
    """

    components.html(game_html, height=680, scrolling=False)

    st.divider()
    st.markdown(
        """
        **Controls:**
        - **SPACE** or **Click** → Flap (make the bird jump up)
        - The bird falls due to gravity — keep flapping to stay airborne
        - Fly through the **gaps** between the green pipes
        - Each pipe passed = **+1 point**
        - Hit a pipe or the ground = **Game Over**
        - Click or press SPACE to restart after dying
        """
    )

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "Results":
    st.title("📊 Training Results")

    genome_path = os.path.join(PROJECT_DIR, "outputs", "best_genome.pkl")

    if not os.path.exists(genome_path):
        st.warning(
            "No training results yet.\n\n"
            "Train locally first:\n"
            "```bash\n"
            "pip install pygame neat-python\n"
            "python3 train.py --mode train --generations 30\n"
            "```\n\n"
            "Once trained, the results will appear here."
        )

        st.divider()
        st.subheader("What You'll See After Training")
        st.markdown(
            """
            After training completes, this page shows:
            - **Best genome fitness score**
            - **Network structure** (nodes, connections evolved by NEAT)
            - **Node genes** and **Connection genes** tables
            - Recommendations for improving performance
            """
        )
        st.stop()

    st.success("Trained model found!")

    # Load genome info
    try:
        import pickle
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)

        st.subheader("Best Genome")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fitness", f"{genome.fitness:.1f}")
        with col2:
            num_connections = len([c for c in genome.connections.values() if c.enabled])
            st.metric("Active Connections", num_connections)
        with col3:
            num_nodes = len(genome.nodes)
            st.metric("Nodes", num_nodes)

        st.divider()

        st.subheader("Network Structure")
        st.markdown(
            f"""
            The evolved neural network has:
            - **5** input neurons (bird state + pipe info)
            - **{num_nodes - 6}** hidden neurons (evolved by NEAT)
            - **1** output neuron (flap decision)
            - **{num_connections}** active connections

            NEAT started with zero hidden nodes and evolved this structure
            through mutation and selection over multiple generations.
            """
        )

        st.divider()

        st.subheader("Genome Details")
        st.markdown("**Node Genes:**")
        node_data = []
        for key, node in genome.nodes.items():
            node_data.append({
                "Node ID": key,
                "Activation": node.activation,
                "Bias": round(node.bias, 3),
            })
        st.dataframe(node_data, use_container_width=True)

        st.markdown("**Connection Genes:**")
        conn_data = []
        for key, conn in genome.connections.items():
            conn_data.append({
                "From": key[0],
                "To": key[1],
                "Weight": round(conn.weight, 3),
                "Enabled": conn.enabled,
            })
        st.dataframe(conn_data, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading genome: {e}")

    st.divider()
    st.subheader("How to Improve")
    st.markdown(
        """
        - **More generations** → More time to evolve complex strategies
        - **Larger population** → More diversity, better exploration
        - **Tweak fitness** → Adjust rewards for different behaviors
        - **Harder pipes** → Smaller gap, faster speed for a tougher challenge
        """
    )
