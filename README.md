# MindSphere Coach

An interactive coaching agent that uses **Active Inference**, **Theory of Mind**, and **empathic planning** to help people identify and overcome personal growth bottlenecks.

Unlike standard coaching chatbots, MindSphere maintains a computational model of the user's mind — it infers *why* a recommendation will or won't work, and adapts in real time.

## How It Works

1. **Calibration** — 10 adaptive questions assess 8 skill dimensions (focus, follow-through, social courage, emotional regulation, systems thinking, self-trust, task clarity, consistency)
2. **Visualization** — an interactive radar chart ("MindSphere") shows your skill profile with dependency edges highlighting bottlenecks
3. **Coaching** — the agent proposes personalized micro-interventions, predicts your response via Theory of Mind, and adapts based on your feedback

The agent is a **companion first, coach second** — it follows your lead, talks about anything, and weaves coaching insights naturally into conversation.

## Architecture

```
Core Engine (Active Inference)
├── Factored POMDP — 12 state factors, mean-field belief updates
├── EFE action selection — pragmatic + epistemic drives
└── Dirichlet parameter learning — A/B matrices refine with experience

Theory of Mind
├── Particle filter — 50 particles over 7-dim user type space
├── Reliability gating — entropy-based confidence threshold
└── Empathic planning — G_social blending with counterfactuals

User Profile (Bayesian Network)
├── LLM-extracted facts — life events, goals, challenges, context
├── Causal edges — "breakup → emotional_stress → focus_impaired"
└── Forward belief propagation — inferred states feed into POMDP

Emotional Inference (Circumplex Model)
├── Predict → Observe → Update loop
├── 8-state POMDP over valence × arousal
└── Prediction error tracking

LLM Layer (Mistral)
├── Dynamic system prompt — beliefs, ToM, emotions, profile injected
├── Natural conversation — companion-style, not scripted
└── Structured extraction — profile facts + causal links via JSON
```

## Quick Start

### Option 1: Double-click (Windows)

Just double-click **`START.bat`** — it checks for Python, installs dependencies, and opens the app in your browser.

### Option 2: Command line

**Prerequisites:** Python 3.10+

```bash
# Clone
git clone https://github.com/mahault/mindsphere-coach.git
cd mindsphere-coach

# Install
pip install -e .

# (Optional) Add your Mistral API key for LLM-powered conversation
cp .env.example .env
# Edit .env and add: MISTRAL_API_KEY=your_key_here

# Run
python scripts/run_demo.py
```

Then open **http://localhost:8000** in your browser.

### API Key

The app works without an API key (template-based responses), but for natural LLM-powered conversation you need a [Mistral API key](https://console.mistral.ai/):

1. Copy `.env.example` to `.env`
2. Replace `your_mistral_api_key_here` with your actual key

## Project Structure

```
src/mindsphere/
├── core/           # Active Inference engine
│   ├── agent.py        # Main orchestrator (~2100 lines)
│   ├── model.py        # Factored POMDP: A/B/C/D matrices
│   ├── inference.py    # Belief updates, EFE computation
│   ├── user_profile.py # Bayesian network user model
│   ├── emotional_state.py  # Circumplex emotion POMDP
│   ├── learning.py     # Dirichlet parameter learning
│   └── dependency_graph.py # Skill dependency DAG
│
├── tom/            # Theory of Mind module
│   ├── particle_filter.py  # User type inference
│   ├── empathy_planner.py  # G_social blending
│   └── trust.py            # Reliability gating
│
├── llm/            # Mistral LLM layer
│   ├── client.py       # API client
│   ├── generator.py    # Conversational engine
│   └── classifier.py   # Structured classification
│
├── content/        # Questions & interventions
├── api/            # FastAPI + WebSocket backend
├── viz/            # Plotly chart generation
└── frontend/       # Vanilla JS + Plotly SPA
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

102 tests covering the POMDP model, inference, particle filter, empathy planner, emotional state, and full integration pipeline.

## License

MIT
