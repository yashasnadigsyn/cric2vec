# Cric2Vec: Deep Player Representations within Cricket

**Cric2Vec** applies the distributional hypothesis from Natural Language Processing to the game of cricket. By treating a match as a semantic sequence of events (ball-by-ball), we learn dense vector representations for players, teams, and venues. 

These embeddings capture latent traits—like "playing style", "role", and "matchup compatibility"—without being explicitly told what a "googly" or a "cover drive" is.

## 🚀 Key Features

- **Context-Aware Embeddings**: `Cricket2VecV2` incorporates Team and Venue context to refine player vectors.
- **Matchup Simulator**: Predict the outcome of any batter vs. bowler duel (dot, run, wicket).
- **Sematic Algebra**: Perform vector arithmetic (e.g., `Kohli - Anchor + Finisher`).
- **Bunny Finder**: Identify batters who are statistically most vulnerable to a specific bowler.
- **Dream XI Generator**: Construct diverse teams using geometric sampling in the embedding space.

## 🛠️ Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

## 📖 Usage

### Inference Engine
Interact with the trained model using the inference script:

```bash
uv run python inference.py
```

### Web Interface
Explore the visual blog and interactive demos:

```bash
# Serve the web directory
cd web
python3 -m http.server
```

## 🧠 Architecture

The model uses a dual-embedding architecture (Batter + Bowler) fused with a Context Encoder (Team + Venue + Match State). These are passed through a dense interaction network to predict the probability distribution over 12 possible ball outcomes.

## 📊 Status

- **Model Accuracy**: ~39% (Predicting exact outcome per ball)
- **Current Version**: V2 (Context-Aware)
- **Data**: IPL Ball-by-Ball dataset (matches up to 2024).

---
*Disclaimer: This project is for research and educational purposes only.*
