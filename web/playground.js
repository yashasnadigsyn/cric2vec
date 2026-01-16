/**
 * Cric2Vec Playground - Interactive API Client
 */

const API_BASE = 'http://localhost:8000';

// --- Player Search ---

async function searchPlayers() {
    const query = document.getElementById('player-search-input').value.trim();
    const resultsDiv = document.getElementById('search-results');

    if (!query) {
        resultsDiv.innerHTML = '<p class="hint">Enter a player name to search</p>';
        return;
    }

    resultsDiv.innerHTML = '<p class="loading">Searching...</p>';

    try {
        const res = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}`);
        if (!res.ok) throw new Error('Search failed');

        const matches = await res.json();

        if (matches.length === 0) {
            resultsDiv.innerHTML = '<p class="hint">No players found</p>';
            return;
        }

        resultsDiv.innerHTML = matches.map(m =>
            `<div class="player-match" onclick="selectPlayer('${m.name}')">
                <span class="player-name">${m.name}</span>
                <span class="match-score">${m.score}%</span>
            </div>`
        ).join('');
    } catch (e) {
        resultsDiv.innerHTML = `<p class="error">Error: ${e.message}. Is the API server running?</p>`;
    }
}

function selectPlayer(name) {
    document.getElementById('player-search-input').value = name;
    document.getElementById('search-results').innerHTML =
        `<p class="hint">Selected: <strong>${name}</strong></p>`;
}

// --- Simulation ---

async function runSimulation() {
    const batter = document.getElementById('sim-batter').value.trim();
    const bowler = document.getElementById('sim-bowler').value.trim();
    const outputDiv = document.getElementById('sim-output');

    if (!batter || !bowler) {
        outputDiv.innerHTML = '<p class="hint">Enter both batter and bowler names</p>';
        return;
    }

    outputDiv.innerHTML = '<p class="loading">Simulating...</p>';

    try {
        const res = await fetch(`${API_BASE}/api/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ batter, bowler })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Simulation failed');
        }

        const data = await res.json();

        let html = `<div class="sim-header">${data.batter} vs ${data.bowler}</div>`;
        html += '<div class="sim-balls">';
        data.balls.forEach((ball, i) => {
            const ballClass = ball.startsWith('w_') ? 'wicket' :
                ball.includes('6') ? 'six' :
                    ball.includes('4') ? 'four' : 'normal';
            html += `<span class="ball ${ballClass}">${i + 1}: ${formatOutcome(ball)}</span>`;
        });
        html += '</div>';
        html += `<div class="sim-summary">${data.total_runs}/${data.wickets}</div>`;

        outputDiv.innerHTML = html;
    } catch (e) {
        outputDiv.innerHTML = `<p class="error">Error: ${e.message}</p>`;
    }
}

function formatOutcome(outcome) {
    const labels = {
        '0_run': 'Dot',
        '1_run': '1 run',
        '2_3_run': '2 runs',
        '4_run': 'FOUR!',
        '6_run': 'SIX!',
        'extras': 'Extra',
        'w_caught': '🏏 Caught!',
        'w_bowled': '🏏 Bowled!',
        'w_lbw': '🏏 LBW!',
        'w_runout': '🏏 Run Out!',
        'w_stumped': '🏏 Stumped!',
        'w_other': '🏏 OUT!'
    };
    return labels[outcome] || outcome;
}

// --- Bunny Finder ---

async function findBunnies() {
    const bowler = document.getElementById('bunny-bowler').value.trim();
    const outputDiv = document.getElementById('bunny-output');

    if (!bowler) {
        outputDiv.innerHTML = '<p class="hint">Enter a bowler name</p>';
        return;
    }

    outputDiv.innerHTML = '<p class="loading">Finding bunnies...</p>';

    try {
        const res = await fetch(`${API_BASE}/api/bunnies`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bowler, top_k: 5 })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Search failed');
        }

        const bunnies = await res.json();

        if (bunnies.length === 0) {
            outputDiv.innerHTML = '<p class="hint">No bunnies found</p>';
            return;
        }

        let html = `<div class="bunny-header">Bunnies for ${bowler}</div>`;
        html += '<div class="bunny-list">';
        bunnies.forEach((b, i) => {
            const pct = (b.wicket_prob * 100).toFixed(1);
            html += `<div class="bunny-item">
                <span class="rank">${i + 1}.</span>
                <span class="name">${b.batter}</span>
                <span class="prob">${pct}%</span>
            </div>`;
        });
        html += '</div>';

        outputDiv.innerHTML = html;
    } catch (e) {
        outputDiv.innerHTML = `<p class="error">Error: ${e.message}</p>`;
    }
}

// --- Probability Visualizer ---

async function showProbabilities() {
    const batter = document.getElementById('prob-batter').value.trim();
    const bowler = document.getElementById('prob-bowler').value.trim();
    const outputDiv = document.getElementById('prob-output');

    if (!batter || !bowler) {
        outputDiv.innerHTML = '<p class="hint">Enter both batter and bowler names</p>';
        return;
    }

    outputDiv.innerHTML = '<p class="loading">Loading probabilities...</p>';

    try {
        const res = await fetch(`${API_BASE}/api/matchup?batter=${encodeURIComponent(batter)}&bowler=${encodeURIComponent(bowler)}`);

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to get probabilities');
        }

        const data = await res.json();
        const probs = data.probabilities;

        // Group outcomes for cleaner display
        const grouped = {
            'Dot Ball': probs['0_run'] || 0,
            'Single': probs['1_run'] || 0,
            '2-3 Runs': probs['2_3_run'] || 0,
            'FOUR': probs['4_run'] || 0,
            'SIX': probs['6_run'] || 0,
            'Extras': probs['extras'] || 0,
            'Wicket': (probs['w_caught'] || 0) + (probs['w_bowled'] || 0) +
                (probs['w_lbw'] || 0) + (probs['w_runout'] || 0) +
                (probs['w_stumped'] || 0) + (probs['w_other'] || 0)
        };

        const colors = {
            'Dot Ball': '#6b7280',
            'Single': '#3b82f6',
            '2-3 Runs': '#8b5cf6',
            'FOUR': '#10b981',
            'SIX': '#7c3aed',
            'Extras': '#f59e0b',
            'Wicket': '#ef4444'
        };

        let html = `<div class="prob-header">${data.batter} vs ${data.bowler}</div>`;
        html += '<div class="prob-bars">';

        for (const [label, prob] of Object.entries(grouped)) {
            const pct = (prob * 100).toFixed(1);
            const barWidth = Math.max(prob * 100, 2); // min 2% width for visibility
            html += `
                <div class="prob-row">
                    <span class="prob-label">${label}</span>
                    <div class="prob-bar-container">
                        <div class="prob-bar" style="width: ${barWidth}%; background: ${colors[label]}"></div>
                    </div>
                    <span class="prob-value">${pct}%</span>
                </div>`;
        }
        html += '</div>';

        outputDiv.innerHTML = html;
    } catch (e) {
        outputDiv.innerHTML = `<p class="error">Error: ${e.message}</p>`;
    }
}

// --- Init ---

document.addEventListener('DOMContentLoaded', () => {
    // Add event listeners
    const searchBtn = document.getElementById('search-btn');
    if (searchBtn) searchBtn.addEventListener('click', searchPlayers);

    const simBtn = document.getElementById('sim-btn');
    if (simBtn) simBtn.addEventListener('click', runSimulation);

    const probBtn = document.getElementById('prob-btn');
    if (probBtn) probBtn.addEventListener('click', showProbabilities);

    // Enter key support
    const searchInput = document.getElementById('player-search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchPlayers();
        });
    }
});
