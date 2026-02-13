# MindSphere Coach — Roadmap

## Vision

MindSphere Coach is an interactive coaching agent that uses Active Inference, Theory of Mind, and empathic planning to help people identify and overcome their personal growth bottlenecks. Unlike standard coaching chatbots, it maintains a computational model of the user's mind and predicts how they'll respond to different interventions — then adapts in real time.

The core innovation: **the agent doesn't just recommend; it infers why a recommendation will or won't work, and adapts.**

---

## Phase 0 — MVP-0+1: Working Demo (Current Build)

**Status:** In Progress
**Timeline:** ~5 days
**Goal:** A functional 5-8 minute coaching experience demonstrable to stakeholders

### What's Built

- **Active Inference Core**
  - Factored POMDP with 12 state factors (8 skills, 2 preferences, 2 friction)
  - Mean-field belief updates (numpy, no JAX dependency)
  - Expected Free Energy (EFE) for action selection (pragmatic + epistemic)
  - Adaptive question ordering by information gain

- **Theory of Mind Module**
  - Particle filter over 7-dimensional user type space (50 particles)
  - 6 seed archetypes: perfectionist, novelty explorer, overwhelmed achiever, autonomous thinker, structure seeker, avoidant
  - Entropy-based reliability gating (sigmoid confidence threshold)
  - Gated prediction: `q_gated = r * q_learned + (1-r) * q_prior`

- **Empathic Planning**
  - Social EFE: `G_social = (1-lambda_eff) * G_system + lambda_eff * G_user`
  - Lambda scales with reliability (cautious when uncertain)
  - Counterfactual computation (gentle vs push comparison)
  - Adjustable empathy dial (0=challenging, 1=gentle)

- **Calibration Interview**
  - 10 questions (8 MC + 2 free text)
  - Per-question A-matrices for Bayesian skill inference
  - Adaptive ordering by expected information gain

- **Skill Dependency Graph**
  - 6 hand-designed dependency edges
  - Bottleneck detection + impact ranking
  - Human-readable explanations

- **LLM Integration (Mistral)**
  - SphereClassifier: MC answer, free text, user choice classification
  - CoachGenerator: LLM-powered conversational engine with dynamic context injection
  - All post-calibration responses routed through Mistral (`mistral-medium-latest`)
  - Web search via Mistral `web_search` tool (current events, facts)
  - Fallback to template responses when no API key available

- **Companion-First Design**
  - Agent follows user's lead — can talk about anything, not just coaching
  - Cognitive load inference: detects disengagement, overwhelm, off-topic intent
  - Continuous cognitive modeling: emotions, interests, engagement tracked per turn
  - Soft ToM updates from conversational signals (not just explicit choices)
  - LLM system prompt dynamically includes: skill profile, ToM predictions, cognitive load, inferred emotions/topics
  - Coaching probes (32 questions) and exercises (24 templates) for when coaching IS wanted

- **POMDP Parameter Learning** (Dirichlet Concentration Parameters)
  - A-matrices learn from observations: each obs increments pseudo-counts
  - B-matrices learn transitions for friction factors (overwhelm, autonomy)
  - Soft-assignment: updates weighted by current belief (expected sufficient statistics)
  - Prior strength controls plasticity (high = stable, low = fast adaptation)
  - Learning summary exposed in belief summary for monitoring
  - Reference: Friston et al. (2016) "Active Inference and Learning"

- **Circumplex Emotional Inference** (Pattisapu & Albarracin 2024)
  - Full predict-observe-update loop for emotional state
  - Arousal = H[Q(s|o)] = posterior entropy (uncertainty → activation)
  - Valence = U - EU = reward prediction error (better/worse than expected)
  - 8 discrete emotions mapped on circumplex: happy, excited, alert, angry, sad, depressed, calm, relaxed
  - POMDP A-matrices for valence/arousal observation models
  - B-matrices with emotional inertia + drift toward neutral
  - Agent PREDICTS emotional state via ToM → LLM classifies text → prediction error drives learning
  - Prediction errors soft-update ToM particles (e.g., negative surprise → lower overwhelm threshold)
  - Heuristic fallback classifier when no LLM available
  - Emotional state + prediction errors injected into LLM system prompt
  - Emotional trajectory tracking (last 5 states, predictions, errors)

- **Web Interface**
  - FastAPI backend with REST endpoints
  - Interactive Plotly radar chart ("dented sphere")
  - Chat UI with MC buttons, choice buttons, and persistent text input
  - Counterfactual side-by-side display
  - Live belief/profile panel with "Model Confidence" explanation
  - Empathy dial slider
  - Typing indicator during processing

- **Safety Features**
  - Autonomy protection defaults (empathy=0.5)
  - "If you feel coerced, I'm miscalibrated" safety check
  - Transparent ToM predictions (shown to user)

### Key Metrics
- Calibration: 10 questions in ~2-3 minutes
- Sphere generation: < 1 second
- Intervention proposal with counterfactual: < 2 seconds
- Full demo: 5-8 minutes
- 57+ tests passing across 6 test modules

---

## Phase 1 — POMDP Cognitive Model + Session Persistence

**Timeline:** 1-2 weeks after Phase 0
**Goal:** Formal POMDP-based cognitive modeling and cross-session memory

### Planned Features

- **Cognitive Load as POMDP Factor**
  - Currently: heuristic keyword detection + ToM dimensions (rule-based)
  - Target: Add `cognitive_load` as a formal state factor in `SphereModel` with:
    - States: `low`, `moderate`, `high`, `overwhelmed` (4 levels)
    - A-matrix: observation model mapping conversation signals to load states
    - B-matrix: transition dynamics (load tends to decay toward moderate, spikes with complex tasks)
    - Bayesian belief updates each turn from classified observations
  - This makes cognitive load inference proper Active Inference, not heuristics

- **Emotional State as Full POMDP Factor**
  - Currently: Circumplex model with predict-observe-update loop (separate from main POMDP)
  - Target: Integrate `emotional_state` into `SphereModel` as a formal state factor
    - Merge valence/arousal beliefs into the factored POMDP
    - EFE computation considers emotional state alongside skill progress
    - Action selection naturally balances user welfare with coaching goals
  - The predict-observe-update loop already provides the core machinery

- **Engagement as POMDP Factor**
  - Add `engagement` as hidden factor: `disengaged`, `passive`, `active`, `flow`
  - Observations from: message length, response time, topic continuity
  - Agent selects actions that maximize engagement alongside coaching progress

- **Unified Cognitive Model**
  - All three new factors (cognitive_load, emotional_state, engagement) integrated into the factored POMDP
  - EFE computation now considers: skill progress + user welfare + cognitive fit
  - Action selection naturally balances coaching with companionship

- **Blindspot Detector**
  - Model of user's self-model vs inferred model
  - Detect discrepancies: "You rate yourself high on focus, but your patterns suggest otherwise"
  - Gentle surfacing with empathic framing

- **Sophisticated Multi-Step Planning**
  - Adapt `SophisticatedPlanner` from empathy project
  - Multi-step policy rollout with social EFE
  - Horizon H=3 lookahead for intervention sequences

- **Richer User Type Dimensions**
  - Add dimensions: perfectionism, time_orientation, social_comparison, growth_mindset
  - Learn from more diverse observation signals (not just explicit choices)
  - Cross-session type refinement

- **Session Persistence (SQLite)**
  - Store session state, beliefs, and conversation history
  - Resume sessions across visits
  - Track progress over time (sphere evolution)

- **Conversation Memory**
  - LLM-generated session summaries
  - Reference previous sessions in conversation
  - Progressive belief refinement across sessions

### Technical Debt
- Unit test coverage > 80%
- CI/CD pipeline
- Environment variable validation
- Error handling audit

---

## Phase 2 — Production Hardening

**Timeline:** 2-3 weeks after Phase 1
**Goal:** Multi-user deployment with proper infrastructure

### Planned Features

- **Authentication**
  - OAuth2 / JWT-based user auth
  - User profiles with session history
  - Data privacy controls

- **Database Migration**
  - PostgreSQL for session and user data
  - Alembic migrations
  - Encrypted belief storage

- **Frontend Framework**
  - Migrate from vanilla JS to React or Svelte
  - Component-based architecture
  - Mobile-responsive design
  - PWA support

- **Analytics Dashboard**
  - Coaching outcomes over time (sphere evolution)
  - Engagement metrics (session completion, intervention acceptance)
  - A/B testing framework for intervention strategies
  - Aggregate anonymized insights

- **Deployment**
  - Docker containerization
  - Cloud deployment (Railway / Fly.io / AWS)
  - Health checks and monitoring
  - Rate limiting

---

## Phase 3 — Advanced Features

**Timeline:** 1-2 months after Phase 2
**Goal:** Research-grade capabilities and integrations

### Planned Features

- **JAX-Accelerated Inference**
  - Port core inference to JAX (reuse `jax_si_empathy_lava.py` patterns)
  - JIT-compiled belief updates and EFE computation
  - 30-100x speedup for real-time multi-step planning

- **Hierarchical Planning**
  - Zone-based planning (reuse `hierarchical_planner.py` from Alignment-experiments)
  - Goal decomposition: life areas → skill areas → micro-steps
  - Multi-scale intervention planning

- **Group Coaching**
  - Multi-agent coordination (reuse multi-agent ToM from empathy project)
  - Team sphere visualization
  - Complementary strength matching
  - Group dynamics modeling

- **External Integrations**
  - Calendar API (schedule interventions)
  - Habit tracker API (verify completion)
  - Slack/Teams notifications
  - Export sphere data as PDF/image

- **Fine-Tuned LLM**
  - Domain-specific fine-tuning on coaching transcripts
  - Better classification accuracy
  - More nuanced empathic responses

---

## Phase 4 — Research Extensions

**Timeline:** 2-4 months after Phase 3
**Goal:** Novel contributions to computational coaching and AI alignment

### Research Directions

- **Recursive Theory of Mind**
  - Depth-2+ ToM: "I think you think I think..."
  - Anticipatory coaching: predict user's self-corrections
  - Reuse depth-K recursive structure from `tom/planning/si_tom.py`

- **Asymmetric Empathy Studies**
  - Translate IPD findings: asymmetry enables coordination
  - Measure coaching effectiveness vs empathy settings
  - Identify optimal lambda trajectories per user type

- **Exploitation Detection**
  - From IPD: empathic structure shapes long-run dynamics
  - Detect when coaching system is being gamed
  - Detect when system is inadvertently manipulative
  - Red-teaming framework for coaching AI safety

- **Active Learning of A/B Matrices**
  - Learn observation models from user data
  - Bayesian model selection for state space structure
  - Adaptive questionnaire design (not just ordering)

- **Publication Targets**
  - "Active Inference for Personalized Coaching: A Theory of Mind Approach"
  - "Empathy-Gated Planning in Human-AI Coaching Interactions"
  - "Reliability-Aware Theory of Mind for Adaptive Interventions"
  - Venues: AAMAS, NeurIPS workshop on AI for social good, CHI

---

## Architecture Lineage

MindSphere Coach builds directly on three existing research codebases:

| Component | Source Project | Key Reuse |
|-----------|---------------|-----------|
| Social EFE, empathy dial | `empathy-prisonner-dilemma` | G_social formula, lambda blending |
| Particle filter, reliability gate | `empathy-prisonner-dilemma` | Weight update, entropy gating, sigmoid trust |
| POMDP coaching structure | `NEXT-prototype` | A/B/C/D matrices, belief update, LLM integration |
| Circumplex emotions | `Alignment-experiments` | Predict-observe-update, emotional_state.py |
| JAX acceleration (future) | `Alignment-experiments` | JIT compilation, vmap batching |
| Multi-agent ToM (future) | `tom` | Recursive planning, perspective-taking |

---

## Dependencies

### Current (Phase 0)
- Python 3.10+
- numpy, fastapi, uvicorn, plotly, pydantic, requests

### Future
- JAX (Phase 3)
- SQLAlchemy + PostgreSQL (Phase 2)
- React/Svelte (Phase 2)
- Docker (Phase 2)
