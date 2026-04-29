"""Streamlit dashboard for Flappy Bird AI project."""

from __future__ import annotations

import os
import subprocess
import pickle
import json

import streamlit as st
import numpy as np

# Resolve project directory reliably (PROJECT_DIR can be empty in Streamlit)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    stats_path = os.path.join(PROJECT_DIR, "outputs", "training_stats.json")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
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
# PLAY GAME
# ═══════════════════════════════════════════════════════════════
elif page == "Play Game":
    st.title("🎮 Play Flappy Bird")

    play_mode = st.radio("Mode", ["🤖 Watch AI Play", "👤 Play Yourself"])

    genome_path = os.path.join(PROJECT_DIR, "outputs", "best_genome.pkl")
    config_path = os.path.join(PROJECT_DIR, "neat_config.txt")

    if play_mode == "🤖 Watch AI Play":
        st.markdown("Watch the trained AI play Flappy Bird perfectly.")

        if not os.path.exists(genome_path):
            st.warning("No trained AI found. Go to **Train AI** tab first!")
        else:
            st.success(f"Trained genome found: `{genome_path}`")
            if st.button("▶️ Watch AI Play", type="primary", use_container_width=True):
                st.info("Opening Pygame window... Press **ESC** to close, **R** to restart.")
                cmd = (
                    f"cd {PROJECT_DIR} && "
                    f"python3 train.py --mode replay --genome-path {genome_path}"
                )
                subprocess.Popen(cmd, shell=True)

    else:
        st.markdown(
            "Play Flappy Bird yourself!\n\n"
            "**Controls:**\n"
            "- **SPACE** → Flap\n"
            "- **ESC** → Quit\n"
            "- **SPACE** (after game over) → Restart\n\n"
            "Close the Pygame window or press ESC when done to see your results here."
        )
        scores_file = os.path.join(PROJECT_DIR, "outputs", "play_scores.json")

        if st.button("▶️ Start Game", type="primary", use_container_width=True):
            st.info("Opening Pygame window... Play and close when done!")
            cmd = (
                f"cd {PROJECT_DIR} && python3 train.py --mode play "
                f"--genome-path {os.path.join(PROJECT_DIR, 'outputs', 'best_genome.pkl')}"
            )
            with st.spinner("Game in progress — close the Pygame window to see results..."):
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)

            # Show results after game closes
            if os.path.exists(scores_file):
                with open(scores_file, "r") as f:
                    scores_data = json.load(f)

                st.balloons()
                st.success("Game session complete!")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🏆 Top Score", scores_data["top_score"])
                with col2:
                    st.metric("🎮 Games Played", scores_data["games_played"])
                with col3:
                    st.metric("📊 Avg Score", scores_data["average_score"])
                with col4:
                    mins = int(scores_data["total_time_seconds"] // 60)
                    secs = int(scores_data["total_time_seconds"] % 60)
                    st.metric("⏱️ Time Played", f"{mins}m {secs}s")

                st.divider()
                st.subheader("Score History")

                all_scores = scores_data["scores"]
                # Bar chart of scores per game
                import pandas as pd
                df = pd.DataFrame({
                    "Game": [f"Game {i+1}" for i in range(len(all_scores))],
                    "Score": all_scores,
                })
                st.bar_chart(df.set_index("Game"))

                # Score breakdown
                st.markdown("**All Scores:**")
                for i, s in enumerate(all_scores):
                    medal = "🥇" if s == max(all_scores) else "🎮"
                    st.markdown(f"{medal} Game {i+1}: **{s}** pipes")
            else:
                st.warning("No scores recorded. Play at least one game and close the window.")

        # Always show previous results if available
        elif os.path.exists(scores_file):
            st.divider()
            st.subheader("📋 Previous Session Results")
            with open(scores_file, "r") as f:
                prev_data = json.load(f)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏆 Top Score", prev_data["top_score"])
            with col2:
                st.metric("🎮 Games Played", prev_data["games_played"])
            with col3:
                st.metric("📊 Avg Score", prev_data["average_score"])

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "Results":
    st.title("📊 Training Results")

    genome_path = os.path.join(PROJECT_DIR, "outputs", "best_genome.pkl")

    if not os.path.exists(genome_path):
        st.warning("No training results yet. Go to **Train AI** to start training!")
        st.stop()

    st.success("Trained model found!")

    # Load genome info
    try:
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
