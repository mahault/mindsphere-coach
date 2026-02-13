/**
 * MindSphere Coach — Frontend Application
 *
 * Handles REST API communication, Plotly radar chart rendering,
 * chat UI, and real-time belief/sphere updates.
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
    beliefContent: document.getElementById('belief-content'),
    tomStats: document.getElementById('tom-stats'),
    safetyNotice: document.getElementById('safety-notice'),
};

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

    try {
        const body = {
            user_message: content,
            message_type: extra.message_type || 'text',
            ...extra,
        };

        const resp = await fetch(`/api/session/${state.sessionId}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();

        // Remove typing indicator
        typingDiv.remove();

        // Process response
        if (data.message) {
            addMessage('assistant', data.message);
        }

        state.phase = data.phase;
        updatePhaseUI(data.phase, data.progress);

        // Show sphere ONLY after calibration is done
        if (data.sphere_data && data.phase !== 'calibration') {
            state.sphereData = data.sphere_data;
            renderSphere(data.sphere_data);
        }

        // Show question if present (calibration phase)
        if (data.question) {
            showQuestion(data.question);
            return;  // showQuestion handles its own input visibility
        }

        // Show counterfactual + intervention (planning/update phase)
        if (data.counterfactual) renderCounterfactual(data.counterfactual);
        if (data.intervention) renderIntervention(data.intervention);

        // Show belief summary
        if (data.belief_summary && data.phase !== 'calibration') {
            renderBeliefUpdate({ beliefs: data.belief_summary });
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

    } catch (err) {
        typingDiv.remove();
        addMessage('system', 'Failed to send message.');
        els.chatInputArea.classList.remove('hidden');
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
// UI RENDERING
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
    }
    if (phase === 'planning' || phase === 'update') {
        els.planPanel.classList.remove('hidden');
    }
    if (phase === 'coaching') {
        // In coaching phase, hide the plan panel (we're past that) and just show the chat
        els.planPanel.classList.add('hidden');
    }
}

function renderSphere(sphereData) {
    els.spherePanel.classList.remove('hidden');

    const categories = sphereData.categories;
    const bottlenecks = sphereData.bottlenecks || [];

    const skills = [
        'focus', 'follow_through', 'social_courage', 'emotional_reg',
        'systems_thinking', 'self_trust', 'task_clarity', 'consistency'
    ];
    const labels = {
        focus: 'Focus', follow_through: 'Follow-through',
        social_courage: 'Social Courage', emotional_reg: 'Emotional Regulation',
        systems_thinking: 'Systems Thinking', self_trust: 'Self-Trust',
        task_clarity: 'Task Clarity', consistency: 'Consistency',
    };

    const theta = skills.map(s => labels[s] || s);
    const r = skills.map(s => categories[s] || 50);
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
        skills.forEach(s => {
            if (bnSkills.has(s)) {
                bnTheta.push(labels[s]);
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
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        font: { color: '#a1a1aa', size: 11 },
        margin: { t: 30, b: 30, l: 50, r: 50 },
        height: 380,
    };

    Plotly.newPlot(els.radarChart, traces, layout, { displayModeBar: false, responsive: true });

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

function renderBeliefUpdate(data) {
    if (!data || !data.beliefs) return;
    const beliefs = data.beliefs;

    let html = '';
    for (const [key, val] of Object.entries(beliefs)) {
        if (key === 'tom_reliability' || key === 'user_type') continue;
        if (typeof val === 'object' && val.score !== undefined) {
            html += `<div class="belief-row">
                <span class="belief-label">${key.replace(/_/g, ' ')}</span>
                <span class="belief-value">${val.score}/100</span>
            </div>`;
        } else if (typeof val === 'object' && val.inferred !== undefined) {
            html += `<div class="belief-row">
                <span class="belief-label">${key.replace(/_/g, ' ')}</span>
                <span class="belief-value">${val.inferred} (${Math.round(val.confidence * 100)}%)</span>
            </div>`;
        }
    }
    els.beliefContent.innerHTML = html;

    if (beliefs.tom_reliability !== undefined) {
        const r = Math.round(beliefs.tom_reliability * 100);
        els.tomStats.innerHTML = `
            <div class="belief-row">
                <span class="belief-label">Model Confidence</span>
                <span class="belief-value">${r}%</span>
            </div>
            <div class="belief-note">
                How confident the agent is in its predictions about you.
                Increases as you interact more.
            </div>
        `;
    }
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
