# Cric2Vec: Deep Player Representations in Cricket

**Cric2Vec** applies NLP's distributional hypothesis to cricket. By treating matches as semantic event sequences (ball-by-ball), we learn dense vector representations for players that capture latent traits like "playing style", "aggression", and "matchup compatibility."

> ⚠️ **For Fun Only!** This is NOT for betting. The model is trained for exploring "what-if" scenarios — like *Shreyanka Patil bowling to Virat Kohli* or *AC Kerr vs MS Dhoni*.

## 🚀 Features

- **IPL + WPL Data**: Combined male (IPL) and female (WPL) cricket data for cross-gender fantasy matchups
- **Context-Aware V2 Model**: Incorporates Team, Venue, and Match State context
- **Web Playground**: Interactive simulator with probability visualizations
- **Matchup Simulator**: Predict outcomes for any batter vs bowler duel
- **Player Search**: Fuzzy matching to find players by name

## 🛠️ Installation

```bash
# Uses uv for dependency management
uv sync
```

## 📖 Usage

### Start the API Server
```bash
uv run python api.py
# Server runs at http://localhost:8000
```

### Open the Playground
Open `web/index.html` in your browser (with the API server running).

**Playground Features:**
- 🔍 **Player Search** - Find players by name
- 🎲 **Fantasy Over Simulator** - Simulate matchups that never happened
- 📊 **Probability Breakdown** - Visual bar chart of outcome probabilities

### Inference CLI
```bash
uv run python inference.py
```

## 🧠 Architecture

The `Cricket2VecV2` model uses:
- Dual embeddings (Batter + Bowler)
- Team and Venue embeddings
- Context encoder (over, innings state)
- Dense classifier → 12 outcome classes

## 📊 Data

| Source | Matches | Gender |
|--------|---------|--------|
| IPL | 2008-2024 | Male |
| WPL | 2023-2025 | Female |

**Total Players**: 891 | **Outcomes**: 12

## 🎯 Model Accuracy

~42% accuracy on predicting exact ball outcome (random baseline ~8%).

## 📁 Project Structure

```
├── api.py              # FastAPI backend
├── inference.py        # Inference engine with analysis tools
├── train_v2.py         # V2 model training script
├── model_v2.py         # Context-aware architecture
├── dataset.py          # PyTorch dataset
├── config.py           # Paths and hyperparameters
├── mapping.py          # ID mapping utilities
├── combine_data.py     # IPL + WPL data pipeline
├── web/
│   ├── index.html      # Blog + Playground
│   ├── style.css
│   └── playground.js   # API client
└── checkpoints/        # Trained model weights
```

---

*Research and educational purposes only. Not for betting or commercial use.*
