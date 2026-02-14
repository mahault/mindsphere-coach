/**
 * MindSphere Coach — Frontend Application
 *
 * Handles REST API communication, Plotly radar chart rendering,
 * chat UI, profile visualization panel, and real-time updates.
 */

// =============================================================================
// STATE
// =============================================================================

const state = {
    sessionId: null,
    phase: 'idle',
    currentQuestion: null,
    sphereData: null,
    questionCount: 0,
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const els = {
    chatMessages: document.getElementById('chat-messages'),
    mcOptions: document.getElementById('mc-options'),
    choiceButtons: document.getElementById('choice-buttons'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    chatInputArea: document.getElementById('chat-input-area'),
    phaseLabel: document.getElementById('phase-label'),
    progressFill: document.getElementById('progress-fill'),
    empathySlider: document.getElementById('empathy-slider'),
    empathyValue: document.getElementById('empathy-value'),
    spherePanel: document.getElementById('sphere-panel'),
    radarChart: document.getElementById('radar-chart'),
    bottleneckInfo: document.getElementById('bottleneck-info'),
    planPanel: document.getElementById('plan-panel'),
    interventionCard: document.getElementById('intervention-card'),
    counterfactualDisplay: document.getElementById('counterfactual-display'),
    profilePanel: document.getElementById('profile-panel'),
    safetyNotice: document.getElementById('safety-notice'),
};

// =============================================================================
// CONSTANTS
// =============================================================================

const SKILL_LABELS = {
    focus: 'Focus',
    follow_through: 'Follow-through',
    social_courage: 'Social Courage',
    emotional_reg: 'Emotional Reg.',
    systems_thinking: 'Systems Thinking',
    self_trust: 'Self-Trust',
    task_clarity: 'Task Clarity',
    consistency: 'Consistency',
};

const SKILL_ORDER = [
    'focus', 'follow_through', 'social_courage', 'emotional_reg',
    'systems_thinking', 'self_trust', 'task_clarity', 'consistency',
];

const BELIEF_LEVEL_LABELS = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];
const BELIEF_LEVEL_COLORS = [
    'rgba(231, 76, 60, 0.8)',    // Very Low — red
    'rgba(243, 156, 18, 0.8)',   // Low — orange
    'rgba(241, 196, 15, 0.8)',   // Medium — yellow
    'rgba(46, 204, 113, 0.8)',   // High — green
    'rgba(39, 174, 96, 0.8)',    // Very High — dark green
];

const TOM_DIM_LABELS = {
    avoids_evaluation: 'Avoids Evaluation',
    hates_long_tasks: 'Prefers Short Tasks',
    novelty_seeking: 'Novelty Seeking',
    structure_preference: 'Structure Preference',
    external_validation: 'External Validation',
    autonomy_sensitivity: 'Autonomy Sensitivity',
    overwhelm_threshold: 'Overwhelm Threshold',
};

const PLOTLY_DARK = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#a1a1aa', size: 11 },
};

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

// =============================================================================
// SESSION MANAGEMENT
// =============================================================================

async function startSession() {
    try {
        const resp = await fetch('/api/session/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lambda_empathy: 0.5 }),
        });
        const data = await resp.json();
        state.sessionId = data.session_id;
        state.phase = data.phase;
        state.questionCount = 0;

        updatePhaseUI(data.phase);
        addMessage('assistant', data.message);

        if (data.question) {
            showQuestion(data.question);
        }
    } catch (err) {
        addMessage('system', 'Failed to start session. Make sure the server is running.');
    }
}

// =============================================================================
// SENDING MESSAGES
// =============================================================================

async function sendMessage(content, extra = {}) {
    if (!state.sessionId) return;

    addMessage('user', content);

    // Hide interactive elements while processing
    els.mcOptions.classList.add('hidden');
    els.chatInputArea.classList.add('hidden');
    els.choiceButtons.classList.add('hidden');

    // Show typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing';
    typingDiv.textContent = '...';
    els.chatMessages.appendChild(typingDiv);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;

    const body = {
        user_message: content,
        message_type: extra.message_type || 'text',
        ...extra,
    };

    // Use streaming endpoint for all phases
    try {
        const resp = await fetch(`/api/session/${state.sessionId}/step-stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let msgDiv = null;
        let metadata = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            let eventType = null;
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    let data;
                    try { data = JSON.parse(dataStr); } catch { continue; }

                    if (eventType === 'metadata') {
                        metadata = data;
                        // Process metadata immediately (phase, sphere, question, etc.)
                        handleStreamMetadata(metadata);
                    } else if (eventType === 'token') {
                        // First token: replace typing indicator with message div
                        if (!msgDiv) {
                            typingDiv.remove();
                            msgDiv = document.createElement('div');
                            msgDiv.className = 'message assistant';
                            msgDiv.textContent = '';
                            els.chatMessages.appendChild(msgDiv);
                        }
                        msgDiv.textContent += data.text || '';
                        els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
                    } else if (eventType === 'done') {
                        // Remove typing indicator if still present
                        if (!msgDiv && typingDiv.parentNode) {
                            typingDiv.remove();
                        }
                    }
                }
            }
        }

        // Finalize: apply post-stream UI updates
        if (typingDiv.parentNode) typingDiv.remove();
        handleStreamFinalize(metadata);

    } catch (err) {
        if (typingDiv.parentNode) typingDiv.remove();
        // Fallback: try non-streaming endpoint
        try {
            const resp = await fetch(`/api/session/${state.sessionId}/step`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (data.message) addMessage('assistant', data.message);
            handleStreamMetadata(data);
            handleStreamFinalize(data);
        } catch {
            addMessage('system', 'Failed to send message.');
            els.chatInputArea.classList.remove('hidden');
        }
    }
}

function handleStreamMetadata(data) {
    if (!data) return;

    state.phase = data.phase || state.phase;
    updatePhaseUI(data.phase, data.progress);

    // Show message if included in metadata (calibration sends full message here)
    if (data.message) {
        addMessage('assistant', data.message);
    }

    // Show sphere ONLY after calibration is done
    if (data.sphere_data && data.phase !== 'calibration') {
        state.sphereData = data.sphere_data;
        renderSphere(data.sphere_data);
    }

    // Show question if present (calibration phase) — AFTER message so order is correct
    if (data.question) {
        showQuestion(data.question);
    }

    // Show counterfactual + intervention (planning/update phase)
    if (data.counterfactual) renderCounterfactual(data.counterfactual);
    if (data.intervention) renderIntervention(data.intervention);
}

function handleStreamFinalize(data) {
    if (!data) {
        els.chatInputArea.classList.remove('hidden');
        return;
    }

    // If there's a question, showQuestion handles input visibility
    if (data.question) return;

    // Refresh profile panel after calibration
    if (data.phase !== 'calibration') {
        refreshProfilePanel();
    }

    // After calibration: always show text input for chatting
    if (data.phase !== 'calibration') {
        els.chatInputArea.classList.remove('hidden');

        // Only show choice buttons when there's an actual intervention to respond to
        if ((data.phase === 'planning' || data.phase === 'update') && !data.is_complete && data.intervention) {
            els.choiceButtons.classList.remove('hidden');
            els.chatInput.placeholder = 'Or type your thoughts...';
        } else {
            els.chatInput.placeholder = 'Type your response...';
        }

        els.chatInput.focus();
    }

    if (data.is_complete) {
        els.choiceButtons.classList.add('hidden');
        els.safetyNotice.classList.remove('hidden');
        els.chatInput.placeholder = 'Session complete — type to continue chatting...';
    }
}

function sendMCAnswer(index, text) {
    els.mcOptions.classList.add('hidden');
    state.questionCount++;
    sendMessage(text, { answer_index: index, message_type: 'mc_choice' });
}

function sendFreeText() {
    const text = els.chatInput.value.trim();
    if (!text) return;
    els.chatInput.value = '';
    sendMessage(text, { message_type: 'text' });
}

function sendChoice(choice) {
    els.choiceButtons.classList.add('hidden');
    const labels = {
        accept: "I'll do it",
        too_hard: "That's too hard for me right now",
        not_relevant: "That's not relevant to me"
    };
    sendMessage(labels[choice] || choice, { choice: choice, message_type: 'user_choice' });
}

// =============================================================================
// UI RENDERING — Core
// =============================================================================

function addMessage(role, content) {
    if (!content || content.trim() === '') return;
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = content;
    els.chatMessages.appendChild(div);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function showQuestion(question) {
    state.currentQuestion = question;

    // Always display the question text as an assistant message
    if (question.question_text) {
        addMessage('assistant', question.question_text);
    }

    if (question.question_type === 'mc' && question.options) {
        // Show MC buttons, hide text input
        els.mcOptions.innerHTML = '';
        els.mcOptions.classList.remove('hidden');
        els.chatInputArea.classList.add('hidden');
        els.choiceButtons.classList.add('hidden');

        question.options.forEach((opt, i) => {
            const btn = document.createElement('button');
            btn.className = 'mc-btn';
            btn.textContent = opt;
            btn.onclick = () => sendMCAnswer(i, opt);
            els.mcOptions.appendChild(btn);
        });
    } else {
        // Free text — show input, hide MC buttons
        els.mcOptions.classList.add('hidden');
        els.choiceButtons.classList.add('hidden');
        els.chatInputArea.classList.remove('hidden');
        els.chatInput.placeholder = 'Share your thoughts...';
        els.chatInput.focus();
    }
}

function updatePhaseUI(phase, progress) {
    const labels = {
        calibration: 'Calibration',
        visualization: 'Your Sphere',
        planning: 'Your Plan',
        update: 'Refining',
        coaching: 'Coaching',
        complete: 'Complete',
    };
    els.phaseLabel.textContent = labels[phase] || phase;

    if (progress !== undefined && progress !== null) {
        els.progressFill.style.width = `${Math.round(progress * 100)}%`;
    }

    // Show/hide panels based on phase — sphere only after calibration
    if (phase === 'visualization' || phase === 'planning' || phase === 'update' || phase === 'coaching' || phase === 'complete') {
        els.spherePanel.classList.remove('hidden');
        els.profilePanel.classList.remove('hidden');
    }
    if (phase === 'planning' || phase === 'update') {
        els.planPanel.classList.remove('hidden');
    }
    if (phase === 'coaching') {
        // In coaching phase, hide the plan panel (we're past that) and just show the chat
        els.planPanel.classList.add('hidden');
    }
}

// =============================================================================
// UI RENDERING — Sphere (Radar Chart)
// =============================================================================

function renderSphere(sphereData) {
    els.spherePanel.classList.remove('hidden');

    const categories = sphereData.categories;
    const bottlenecks = sphereData.bottlenecks || [];

    const theta = SKILL_ORDER.map(s => SKILL_LABELS[s] || s);
    const r = SKILL_ORDER.map(s => categories[s] || 50);
    theta.push(theta[0]);
    r.push(r[0]);

    const traces = [
        {
            type: 'scatterpolar',
            r: r,
            theta: theta,
            fill: 'toself',
            name: 'Your Sphere',
            line: { color: '#4A90D9', width: 2 },
            fillcolor: 'rgba(74, 144, 217, 0.25)',
            hovertemplate: '%{theta}: %{r:.0f}/100<extra></extra>',
        },
    ];

    if (bottlenecks.length > 0) {
        const bnSkills = new Set(bottlenecks.map(b => b.blocker));
        const bnTheta = [];
        const bnR = [];
        SKILL_ORDER.forEach(s => {
            if (bnSkills.has(s)) {
                bnTheta.push(SKILL_LABELS[s]);
                bnR.push(categories[s] || 50);
            }
        });
        if (bnTheta.length > 0) {
            traces.push({
                type: 'scatterpolar',
                r: bnR,
                theta: bnTheta,
                mode: 'markers',
                name: 'Bottlenecks',
                marker: { color: '#E74C3C', size: 14, symbol: 'diamond' },
                hovertemplate: '%{theta}: %{r:.0f} (bottleneck)<extra></extra>',
            });
        }
    }

    const layout = {
        polar: {
            radialaxis: {
                visible: true, range: [0, 100],
                tickvals: [20, 40, 60, 80, 100],
                gridcolor: 'rgba(200, 200, 200, 0.15)',
            },
            angularaxis: { gridcolor: 'rgba(200, 200, 200, 0.15)' },
            bgcolor: 'rgba(0, 0, 0, 0)',
        },
        showlegend: false,
        ...PLOTLY_DARK,
        margin: { t: 30, b: 30, l: 50, r: 50 },
        height: 380,
    };

    Plotly.newPlot(els.radarChart, traces, layout, PLOTLY_CONFIG);

    if (bottlenecks.length > 0) {
        els.bottleneckInfo.innerHTML = bottlenecks.slice(0, 3).map(bn => {
            const blocker = bn.blocker.replace(/_/g, ' ');
            const blocked = bn.blocked.map(b => b.replace(/_/g, ' ')).join(', ');
            const score = Math.round(bn.score * 100);
            return `<div class="bottleneck-item">
                <strong>${blocker}</strong> (${score}/100) is limiting: ${blocked}
            </div>`;
        }).join('');
    } else {
        els.bottleneckInfo.innerHTML = '<p>No significant bottlenecks. Your sphere is fairly balanced.</p>';
    }
}

// =============================================================================
// UI RENDERING — Plan / Counterfactual
// =============================================================================

function renderCounterfactual(cf) {
    if (!cf) return;
    els.planPanel.classList.remove('hidden');

    const gentle = cf.gentle;
    const push = cf.push;

    els.counterfactualDisplay.innerHTML = `
        <div class="cf-option gentle">
            <div class="cf-label" style="color: var(--accent-green);">Gentle (${gentle.duration_minutes}min)</div>
            <p style="font-size: 12px; margin-bottom: 8px;">${gentle.description}</p>
            <div class="cf-stat">
                <span>Predicted completion</span>
                <span class="cf-stat-value">${Math.round(gentle.p_completion * 100)}%</span>
            </div>
            <div class="cf-stat">
                <span>Dropout risk</span>
                <span class="cf-stat-value">${Math.round(gentle.p_dropout * 100)}%</span>
            </div>
        </div>
        <div class="cf-option push">
            <div class="cf-label" style="color: var(--accent-amber);">Challenging (${push.duration_minutes}min)</div>
            <p style="font-size: 12px; margin-bottom: 8px;">${push.description}</p>
            <div class="cf-stat">
                <span>Predicted completion</span>
                <span class="cf-stat-value">${Math.round(push.p_completion * 100)}%</span>
            </div>
            <div class="cf-stat">
                <span>Dropout risk</span>
                <span class="cf-stat-value">${Math.round(push.p_dropout * 100)}%</span>
            </div>
        </div>
    `;

    const conf = cf.confidence || 0;
    if (conf < 0.3) {
        els.counterfactualDisplay.innerHTML += `
            <p style="grid-column: 1 / -1; font-size: 11px; color: var(--text-secondary); font-style: italic; margin-top: 8px;">
                These predictions are preliminary -- I'm still learning your patterns.
            </p>
        `;
    }
}

function renderIntervention(intervention) {
    if (!intervention) return;
    els.planPanel.classList.remove('hidden');

    els.interventionCard.innerHTML = `
        <div class="intervention-desc">${intervention.description}</div>
        <div class="intervention-meta">
            <span>Target: ${(intervention.target_skill || '').replace(/_/g, ' ')}</span>
            <span>${intervention.duration_minutes} min</span>
            <span>Difficulty: ${Math.round((intervention.difficulty || 0) * 100)}%</span>
        </div>
    `;
}

// =============================================================================
// PROFILE PANEL — Data fetching
// =============================================================================

async function refreshProfilePanel() {
    if (!state.sessionId) return;
    try {
        const resp = await fetch(`/api/session/${state.sessionId}/profile-data`);
        const data = await resp.json();
        renderSkillDistributions(data.skill_beliefs, data.skill_scores, data.score_deltas);
        renderCircumplex(data.emotional_state);
        renderTomProfile(data.tom_profile);
        renderDependencyGraph(data.dependency_graph);
        renderBayesNet(data.profile_facts);
    } catch (err) {
        console.warn('Failed to refresh profile panel:', err);
    }
}

// =============================================================================
// PROFILE PANEL — Skill Belief Distributions
// =============================================================================

function renderSkillDistributions(skillBeliefs, skillScores, scoreDeltas) {
    if (!skillBeliefs) return;
    const container = document.getElementById('skill-dist-chart');
    if (!container) return;

    // Reverse skill order so first skill appears at top
    const skills = [...SKILL_ORDER].reverse();
    const deltas = scoreDeltas || {};

    // Build stacked horizontal bar traces — one per belief level
    const traces = BELIEF_LEVEL_LABELS.map((level, i) => ({
        type: 'bar',
        name: level,
        y: skills.map(s => {
            const label = SKILL_LABELS[s] || s;
            const delta = deltas[s] || 0;
            const arrow = delta > 1 ? ' \u2191' : delta < -1 ? ' \u2193' : '';
            return label + arrow;
        }),
        x: skills.map(s => {
            const belief = skillBeliefs[s];
            return belief ? Math.round(belief[i] * 100) : 0;
        }),
        orientation: 'h',
        marker: { color: BELIEF_LEVEL_COLORS[i] },
        hovertemplate: `%{y}: ${level} = %{x}%<extra></extra>`,
    }));

    const layout = {
        barmode: 'stack',
        ...PLOTLY_DARK,
        margin: { t: 8, b: 30, l: 120, r: 40 },
        height: 280,
        xaxis: {
            title: 'Belief %',
            range: [0, 100],
            gridcolor: 'rgba(200,200,200,0.1)',
            ticksuffix: '%',
        },
        yaxis: {
            automargin: true,
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            x: 0, y: -0.2,
            font: { size: 9, color: '#a1a1aa' },
        },
        bargap: 0.3,
    };

    Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
}

// =============================================================================
// PROFILE PANEL — Emotional Circumplex
// =============================================================================

function renderCircumplex(emotionalState) {
    if (!emotionalState) return;
    const container = document.getElementById('circumplex-chart');
    if (!container) return;

    // Backend arousal is now centered [-0.8, 0.8] — use directly
    const traces = [];

    // Trajectory dots (fading opacity)
    const trajectory = emotionalState.trajectory || [];
    if (trajectory.length > 0) {
        const traj = trajectory.slice(-5);
        traces.push({
            type: 'scatter',
            mode: 'lines+markers',
            x: traj.map(s => s.valence || 0),
            y: traj.map(s => s.arousal || 0),
            marker: {
                color: traj.map((_, i) => `rgba(74, 144, 217, ${0.2 + 0.15 * i})`),
                size: traj.map((_, i) => 6 + i),
            },
            line: { color: 'rgba(74, 144, 217, 0.3)', width: 1, dash: 'dot' },
            name: 'Trajectory',
            hovertemplate: 'V=%{x:.2f}, A=%{y:.2f}<extra>history</extra>',
        });
    }

    // Current position
    const current = emotionalState.current;
    if (current) {
        const v = current.valence || 0;
        const a = current.arousal || 0;
        const emotion = current.emotion || current.emotion_label || '?';
        // Color by quadrant
        let dotColor = '#4A90D9';
        if (v < 0 && a > 0) dotColor = '#E74C3C';       // Tense/Angry
        else if (v > 0 && a > 0) dotColor = '#22c55e';   // Excited/Happy
        else if (v < 0 && a <= 0) dotColor = '#8b5cf6';   // Sad/Bored
        else if (v >= 0 && a <= 0) dotColor = '#06b6d4';   // Calm/Relaxed

        traces.push({
            type: 'scatter',
            mode: 'markers+text',
            x: [v],
            y: [a],
            marker: { color: dotColor, size: 16, line: { color: 'white', width: 2 } },
            text: [emotion],
            textposition: 'top center',
            textfont: { color: '#e4e4e7', size: 11 },
            name: 'Current',
            hovertemplate: `${emotion}<br>Valence: %{x:.2f}<br>Arousal: %{y:.2f}<extra></extra>`,
        });
    }

    const layout = {
        ...PLOTLY_DARK,
        margin: { t: 20, b: 40, l: 45, r: 20 },
        height: 260,
        xaxis: {
            title: 'Valence',
            range: [-1.1, 1.1],
            zeroline: true, zerolinecolor: 'rgba(200,200,200,0.2)',
            gridcolor: 'rgba(200,200,200,0.08)',
        },
        yaxis: {
            title: 'Arousal',
            range: [-1.1, 1.1],
            zeroline: true, zerolinecolor: 'rgba(200,200,200,0.2)',
            gridcolor: 'rgba(200,200,200,0.08)',
        },
        showlegend: false,
        annotations: [
            { x: -0.7, y: 0.8, text: 'Tense', showarrow: false, font: { color: 'rgba(231,76,60,0.5)', size: 10 } },
            { x: 0.7, y: 0.8, text: 'Excited', showarrow: false, font: { color: 'rgba(34,197,94,0.5)', size: 10 } },
            { x: -0.7, y: -0.8, text: 'Sad', showarrow: false, font: { color: 'rgba(139,92,246,0.5)', size: 10 } },
            { x: 0.7, y: -0.8, text: 'Calm', showarrow: false, font: { color: 'rgba(6,182,212,0.5)', size: 10 } },
        ],
        // Draw reference circle
        shapes: [{
            type: 'circle',
            xref: 'x', yref: 'y',
            x0: -1, y0: -1, x1: 1, y1: 1,
            line: { color: 'rgba(200,200,200,0.1)', width: 1 },
        }],
    };

    Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
}

// =============================================================================
// PROFILE PANEL — ToM User Type
// =============================================================================

function renderTomProfile(tomProfile) {
    if (!tomProfile) return;
    const chartContainer = document.getElementById('tom-chart');
    const reliabilityContainer = document.getElementById('tom-reliability');
    if (!chartContainer) return;

    const dims = tomProfile.dimensions || {};
    const dimKeys = Object.keys(TOM_DIM_LABELS);
    const labels = dimKeys.map(k => TOM_DIM_LABELS[k] || k);
    const values = dimKeys.map(k => Math.round((dims[k] || 0.5) * 100));

    // Color gradient: blue (low) → amber (mid) → red (high sensitivity)
    const colors = values.map(v => {
        if (v < 30) return 'rgba(74, 144, 217, 0.8)';
        if (v < 60) return 'rgba(245, 158, 11, 0.8)';
        return 'rgba(231, 76, 60, 0.8)';
    });

    const traces = [{
        type: 'bar',
        y: labels.reverse(),
        x: values.reverse(),
        orientation: 'h',
        marker: { color: colors.reverse() },
        hovertemplate: '%{y}: %{x}%<extra></extra>',
    }];

    const layout = {
        ...PLOTLY_DARK,
        margin: { t: 8, b: 30, l: 130, r: 30 },
        height: 230,
        xaxis: {
            range: [0, 100],
            gridcolor: 'rgba(200,200,200,0.1)',
            ticksuffix: '%',
        },
        yaxis: { automargin: true },
        showlegend: false,
        bargap: 0.3,
    };

    Plotly.newPlot(chartContainer, traces, layout, PLOTLY_CONFIG);

    // Reliability badge
    if (reliabilityContainer) {
        const r = Math.round((tomProfile.reliability || 0) * 100);
        const color = r > 60 ? 'var(--accent-green)' : r > 30 ? 'var(--accent-amber)' : 'var(--accent-red)';
        reliabilityContainer.innerHTML = `
            <div class="tom-reliability-badge">
                Model Confidence: <span style="color: ${color}; font-weight: 600;">${r}%</span>
            </div>
        `;
    }
}

// =============================================================================
// PROFILE PANEL — Causal Dependency Graph
// =============================================================================

function renderDependencyGraph(depGraph) {
    if (!depGraph) return;
    const container = document.getElementById('dependency-chart');
    if (!container) return;

    const nodes = depGraph.nodes || [];
    const edges = depGraph.edges || [];

    if (nodes.length === 0) {
        container.innerHTML = '<p class="profile-empty">No data yet.</p>';
        return;
    }

    // Circular layout for 8 nodes
    const n = nodes.length;
    const cx = 0.5, cy = 0.5, radius = 0.38;
    const positions = {};
    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i / n) - Math.PI / 2;
        positions[node.id] = {
            x: cx + radius * Math.cos(angle),
            y: cy + radius * Math.sin(angle),
        };
    });

    const traces = [];

    // Edge lines
    edges.forEach(edge => {
        const s = positions[edge.source];
        const t = positions[edge.target];
        if (!s || !t) return;

        // Draw line
        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: [s.x, t.x],
            y: [s.y, t.y],
            line: {
                color: `rgba(74, 144, 217, ${0.3 + edge.weight * 0.5})`,
                width: 1 + edge.weight * 3,
            },
            hoverinfo: 'skip',
            showlegend: false,
        });

        // Arrowhead (small triangle at 80% along the line)
        const mx = s.x + 0.75 * (t.x - s.x);
        const my = s.y + 0.75 * (t.y - s.y);
        traces.push({
            type: 'scatter',
            mode: 'markers',
            x: [mx],
            y: [my],
            marker: {
                symbol: 'triangle-up',
                size: 8,
                color: `rgba(74, 144, 217, ${0.4 + edge.weight * 0.4})`,
                angle: Math.atan2(t.y - s.y, t.x - s.x) * 180 / Math.PI - 90,
            },
            hovertemplate: `${(SKILL_LABELS[edge.source] || edge.source)} \u2192 ${(SKILL_LABELS[edge.target] || edge.target)}<br>Weight: ${edge.weight}<extra></extra>`,
            showlegend: false,
        });
    });

    // Nodes
    const nodeX = nodes.map(n => positions[n.id].x);
    const nodeY = nodes.map(n => positions[n.id].y);
    const nodeColors = nodes.map(n => {
        const score = n.score || 50;
        if (n.is_bottleneck) return '#E74C3C';
        if (score < 35) return '#f59e0b';
        if (score < 55) return '#4A90D9';
        return '#22c55e';
    });
    const nodeLabels = nodes.map(n => SKILL_LABELS[n.id] || n.id);
    const nodeSizes = nodes.map(n => n.is_bottleneck ? 20 : 14);

    traces.push({
        type: 'scatter',
        mode: 'markers+text',
        x: nodeX,
        y: nodeY,
        marker: {
            color: nodeColors,
            size: nodeSizes,
            line: { color: nodes.map(n => n.is_bottleneck ? '#E74C3C' : 'rgba(200,200,200,0.3)'), width: 2 },
        },
        text: nodeLabels,
        textposition: nodes.map((_, i) => {
            const angle = (2 * Math.PI * i / n) - Math.PI / 2;
            // Position labels outside the circle
            if (angle > -Math.PI / 4 && angle < Math.PI / 4) return 'top center';
            if (angle >= Math.PI / 4 && angle < 3 * Math.PI / 4) return 'middle right';
            if (angle >= 3 * Math.PI / 4 || angle < -3 * Math.PI / 4) return 'bottom center';
            return 'middle left';
        }),
        textfont: { color: '#e4e4e7', size: 10 },
        hovertemplate: nodes.map(n => {
            const label = SKILL_LABELS[n.id] || n.id;
            const bn = n.is_bottleneck ? ' (BOTTLENECK)' : '';
            return `${label}: ${Math.round(n.score)}/100${bn}<extra></extra>`;
        }),
        showlegend: false,
    });

    const layout = {
        ...PLOTLY_DARK,
        margin: { t: 10, b: 10, l: 10, r: 10 },
        height: 300,
        xaxis: { visible: false, range: [-0.05, 1.05] },
        yaxis: { visible: false, range: [-0.05, 1.05], scaleanchor: 'x' },
        showlegend: false,
    };

    Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
}

// =============================================================================
// PROFILE PANEL — Bayesian Network (Facts + Causal Graph)
// =============================================================================

function renderBayesNet(profileFacts) {
    if (!profileFacts) return;
    const container = document.getElementById('facts-content');
    if (!container) return;

    const facts = profileFacts.facts || [];
    const bayesNet = profileFacts.bayes_net || {};
    const bnNodes = bayesNet.nodes || [];
    const bnEdges = bayesNet.edges || [];

    if (facts.length === 0 && bnNodes.length === 0) {
        container.innerHTML = '<p class="profile-empty">No facts extracted yet. They\'ll appear as we talk.</p>';
        return;
    }

    let html = '';

    // If we have a Bayesian network with edges, render it as a graph
    if (bnNodes.length > 0 && bnEdges.length > 0) {
        html += '<div id="bayes-graph"></div>';
    }

    // Facts list (below graph, if present)
    if (facts.length > 0) {
        const observed = facts.filter(f => f.source === 'explicit');
        const inferred = facts.filter(f => f.source !== 'explicit');

        if (observed.length > 0) {
            html += '<div class="facts-section"><div class="facts-heading">Observed</div>';
            observed.forEach(f => {
                const cat = f.category || 'general';
                const valenceClass = f.valence === 'negative' ? 'neg' : f.valence === 'positive' ? 'pos' : 'neu';
                html += `<div class="fact-item ${valenceClass}">
                    <span class="fact-badge">${cat}</span>
                    <span class="fact-text">${f.content || ''}</span>
                </div>`;
            });
            html += '</div>';
        }

        if (inferred.length > 0) {
            html += '<div class="facts-section"><div class="facts-heading">Inferred</div>';
            inferred.forEach(f => {
                const cat = f.category || 'inferred';
                html += `<div class="fact-item inferred">
                    <span class="fact-badge">${cat}</span>
                    <span class="fact-text">${f.content || ''}</span>
                </div>`;
            });
            html += '</div>';
        }
    } else if (bnNodes.length > 0) {
        // Show nodes as list if no flat facts but we have BN nodes
        html += '<div class="facts-section"><div class="facts-heading">Causal Inferences</div>';
        bnNodes.forEach(n => {
            const prob = Math.round((n.probability || 0) * 100);
            const obsClass = n.observed ? 'pos' : 'inferred';
            html += `<div class="fact-item ${obsClass}">
                <span class="fact-badge">${n.observed ? 'observed' : 'inferred'}</span>
                <span class="fact-text">${n.content || ''}</span>
                <span class="fact-prob-inline">${prob}%</span>
            </div>`;
        });
        html += '</div>';
    }

    container.innerHTML = html;

    // Render the Bayesian network graph if we have edges
    if (bnNodes.length > 0 && bnEdges.length > 0) {
        renderBayesGraph(bnNodes, bnEdges);
    }
}

function renderBayesGraph(nodes, edges) {
    const container = document.getElementById('bayes-graph');
    if (!container) return;

    // Use a top-down layered layout: observed nodes at top, inferred below
    const observed = nodes.filter(n => n.observed);
    const inferred = nodes.filter(n => !n.observed);

    // Position nodes in layers
    const positions = {};
    const width = 1.0;

    // Top layer: observed facts
    observed.forEach((n, i) => {
        positions[n.id] = {
            x: (i + 0.5) / Math.max(observed.length, 1) * width,
            y: 0.85,
        };
    });

    // Bottom layer: inferred states
    inferred.forEach((n, i) => {
        positions[n.id] = {
            x: (i + 0.5) / Math.max(inferred.length, 1) * width,
            y: 0.15,
        };
    });

    const traces = [];

    // Edges
    edges.forEach(edge => {
        const s = positions[edge.source];
        const t = positions[edge.target];
        if (!s || !t) return;

        const isNeg = edge.relationship === 'decreases' || edge.relationship === 'blocks';
        const edgeColor = isNeg
            ? `rgba(231, 76, 60, ${0.3 + edge.strength * 0.5})`
            : `rgba(74, 144, 217, ${0.3 + edge.strength * 0.5})`;

        traces.push({
            type: 'scatter',
            mode: 'lines',
            x: [s.x, t.x],
            y: [s.y, t.y],
            line: { color: edgeColor, width: 1 + edge.strength * 3 },
            hoverinfo: 'skip',
            showlegend: false,
        });

        // Arrow at midpoint
        const mx = s.x + 0.7 * (t.x - s.x);
        const my = s.y + 0.7 * (t.y - s.y);
        traces.push({
            type: 'scatter',
            mode: 'markers',
            x: [mx], y: [my],
            marker: {
                symbol: 'triangle-down',
                size: 7,
                color: edgeColor,
            },
            hovertemplate: `${edge.relationship} (${Math.round(edge.strength * 100)}%)<extra></extra>`,
            showlegend: false,
        });
    });

    // Nodes
    const allNodes = [...observed, ...inferred];
    const nodeX = allNodes.map(n => positions[n.id]?.x || 0.5);
    const nodeY = allNodes.map(n => positions[n.id]?.y || 0.5);
    const nodeColors = allNodes.map(n => {
        if (n.observed) return '#22c55e';
        const p = n.probability || 0;
        if (p > 0.6) return '#E74C3C';
        if (p > 0.3) return '#f59e0b';
        return '#4A90D9';
    });
    const nodeText = allNodes.map(n => {
        const label = (n.content || '').length > 25
            ? (n.content || '').substring(0, 22) + '...'
            : (n.content || '');
        return label;
    });
    const nodeSizes = allNodes.map(n => n.observed ? 14 : 10 + (n.probability || 0) * 8);

    traces.push({
        type: 'scatter',
        mode: 'markers+text',
        x: nodeX, y: nodeY,
        marker: {
            color: nodeColors,
            size: nodeSizes,
            line: { color: 'rgba(200,200,200,0.3)', width: 1.5 },
        },
        text: nodeText,
        textposition: allNodes.map(n => n.observed ? 'top center' : 'bottom center'),
        textfont: { color: '#e4e4e7', size: 9 },
        hovertemplate: allNodes.map(n => {
            const prob = Math.round((n.probability || 0) * 100);
            const type = n.observed ? 'Observed' : 'Inferred';
            return `${n.content}<br>${type} (${prob}%)<extra></extra>`;
        }),
        showlegend: false,
    });

    const layout = {
        ...PLOTLY_DARK,
        margin: { t: 10, b: 10, l: 10, r: 10 },
        height: Math.max(180, 60 + allNodes.length * 20),
        xaxis: { visible: false, range: [-0.1, 1.1] },
        yaxis: { visible: false, range: [-0.1, 1.0] },
        showlegend: false,
        annotations: [
            { x: 0.0, y: 0.95, text: 'Observed', showarrow: false, font: { color: 'rgba(34,197,94,0.6)', size: 9 } },
            { x: 0.0, y: 0.05, text: 'Inferred', showarrow: false, font: { color: 'rgba(74,144,217,0.6)', size: 9 } },
        ],
    };

    Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

els.sendBtn.addEventListener('click', () => {
    sendFreeText();
});

els.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendFreeText();
    }
});

els.empathySlider.addEventListener('input', (e) => {
    const val = (e.target.value / 100).toFixed(2);
    els.empathyValue.textContent = val;

    if (state.sessionId) {
        fetch(`/api/session/${state.sessionId}/empathy-dial`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lambda_value: parseFloat(val) }),
        });
    }
});

document.querySelectorAll('.choice-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        sendChoice(btn.dataset.choice);
    });
});

document.getElementById('safety-btn')?.addEventListener('click', () => {
    els.chatInputArea.classList.remove('hidden');
    sendMessage("I feel like the suggestions aren't quite right for me.", {
        message_type: 'text',
    });
});

// =============================================================================
// LLM STATUS CHECK
// =============================================================================

async function checkLLMStatus() {
    const badge = document.getElementById('llm-status');
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        if (data.llm_available) {
            badge.textContent = 'LLM: Connected';
            badge.className = 'llm-badge connected';
            badge.title = 'Mistral API key loaded — natural conversation mode';
        } else {
            badge.textContent = 'LLM: Template mode';
            badge.className = 'llm-badge template';
            badge.title = 'No API key — using template responses. Add MISTRAL_API_KEY for natural conversation.';
        }
    } catch {
        badge.textContent = 'LLM: Offline';
        badge.className = 'llm-badge template';
        badge.title = 'Could not check LLM status';
    }
}

// =============================================================================
// INIT
// =============================================================================

checkLLMStatus();
startSession();
