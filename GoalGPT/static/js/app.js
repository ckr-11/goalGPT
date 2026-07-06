/* ==========================================================================
   GoalGPT JS Controller
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // State Variables
    let allTeams = [];
    let selectedHome = '';
    let selectedAway = '';
    let predictionHistory = [];

    // DOM Elements — Inputs
    const homeSearch = document.getElementById('home-search');
    const awaySearch = document.getElementById('away-search');
    const homeOptions = document.getElementById('home-options');
    const awayOptions = document.getElementById('away-options');
    const predictionForm = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');

    // DOM Elements — State panels
    const loadingState = document.getElementById('loading-state');
    const errorPanel = document.getElementById('error-panel');
    const errorTitle = document.getElementById('error-title');
    const errorDesc = document.getElementById('error-desc');
    const errorSuggestions = document.getElementById('error-suggestions');
    const predictionDashboard = document.getElementById('prediction-dashboard');

    // Theme Toggle
    const themeToggleBtn = document.getElementById('theme-toggle');
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
    });

    // Only initialize predictor logic if prediction inputs are present on the page
    if (homeSearch && awaySearch) {
        // 1. Initial Load: Fetch Available Teams
    fetch('/api/teams')
        .then(res => res.json())
        .then(data => {
            allTeams = data.teams || [];
            setupSearchDropdown(homeSearch, homeOptions, 'home');
            setupSearchDropdown(awaySearch, awayOptions, 'away');
        })
        .catch(err => {
            showError('Initialization Error', 'Failed to retrieve World Cup teams from the backend prediction engine.');
            console.error(err);
        });

    // Load History from Session
    try {
        const cached = sessionStorage.getItem('goalgpt_history');
        if (cached) {
            predictionHistory = JSON.parse(cached);
            renderHistory();
        }
    } catch (e) {
        console.error(e);
    }

    // 2. Setup Dropdown Autocomplete Search
    function setupSearchDropdown(inputEl, listEl, type) {
        // Toggle list visibility on click
        inputEl.addEventListener('focus', () => {
            closeAllDropdowns();
            renderDropdownList(inputEl.value, listEl, type);
            listEl.classList.remove('hidden');
            inputEl.closest('.search-dropdown-wrapper').classList.add('active');
        });

        inputEl.addEventListener('input', (e) => {
            renderDropdownList(e.target.value, listEl, type);
            listEl.classList.remove('hidden');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!inputEl.parentNode.contains(e.target) && !listEl.contains(e.target)) {
                listEl.classList.add('hidden');
                inputEl.closest('.search-dropdown-wrapper').classList.remove('active');
            }
        });
    }

    function closeAllDropdowns() {
        document.querySelectorAll('.dropdown-list').forEach(el => el.classList.add('hidden'));
        document.querySelectorAll('.search-dropdown-wrapper').forEach(el => el.classList.remove('active'));
    }

    function renderDropdownList(query, listEl, type) {
        listEl.innerHTML = '';
        const searchVal = query.trim().toLowerCase();
        
        // Filter out selected opposite team to prevent choosing the same team twice
        const oppositeTeam = type === 'home' ? selectedAway : selectedHome;

        const filtered = allTeams.filter(team => {
            const matchesQuery = team.toLowerCase().includes(searchVal);
            return matchesQuery;
        });

        if (filtered.length === 0) {
            const emptyItem = document.createElement('div');
            emptyItem.className = 'dropdown-item disabled';
            emptyItem.innerText = 'No teams matched';
            listEl.appendChild(emptyItem);
            return;
        }

        filtered.forEach(team => {
            const item = document.createElement('div');
            item.className = 'dropdown-item';
            if (team === oppositeTeam) {
                item.className += ' disabled';
                item.title = 'Cannot select the same team twice';
            }
            item.innerText = team;

            item.addEventListener('click', () => {
                if (team === oppositeTeam) return;

                if (type === 'home') {
                    selectedHome = team;
                    homeSearch.value = team;
                } else {
                    selectedAway = team;
                    awaySearch.value = team;
                }
                listEl.classList.add('hidden');
                const inputElWrapperActive = inputEl.closest('.search-dropdown-wrapper');
                if (inputElWrapperActive) inputElWrapperActive.classList.remove('active');
            });

            listEl.appendChild(item);
        });
    }

    // 3. Form Submit Prediction
    predictionForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        const homeVal = homeSearch.value.trim();
        const awayVal = awaySearch.value.trim();

        if (!homeVal || !awayVal) {
            showError('Input Missing', 'Both Home Team and Away Team selections are required to generate predictions.');
            return;
        }

        if (homeVal.toLowerCase() === awayVal.toLowerCase()) {
            showError('Invalid Selection', 'Home and Away teams must be different.');
            return;
        }

        // Trigger Loading State
        submitBtn.disabled = true;
        loadingState.classList.remove('hidden');
        errorPanel.classList.add('hidden');
        predictionDashboard.classList.add('hidden');
        closeAllDropdowns();

        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ home_team: homeVal, away_team: awayVal })
        })
        .then(async res => {
            const data = await res.json();
            if (!res.ok) {
                throw data;
            }
            return data;
        })
        .then(data => {
            loadingState.classList.add('hidden');
            submitBtn.disabled = false;
            renderDashboard(data);
            addToHistory(data);
        })
        .catch(err => {
            loadingState.classList.add('hidden');
            submitBtn.disabled = false;
            
            if (err.validation_errors) {
                const primaryErr = err.validation_errors[0];
                showError('Team Not Found', primaryErr.message, primaryErr.suggestions);
            } else {
                showError('Prediction Failure', err.message || 'An unexpected error occurred during prediction generation.');
            }
            console.error(err);
        });
    });

    // 4. Render Dashboard Details
    function renderDashboard(data) {
        const pred = data.prediction;
        const anal = data.analytics;

        const homeName = pred.score_prediction.predicted_scoreline.home_team;
        const awayName = pred.score_prediction.predicted_scoreline.away_team;
        const homeGoals = pred.score_prediction.predicted_scoreline.home_goals;
        const awayGoals = pred.score_prediction.predicted_scoreline.away_goals;

        // Header Headers
        document.getElementById('table-home-hdr').innerText = homeName;
        document.getElementById('table-away-hdr').innerText = awayName;
        document.getElementById('player-home-title').innerText = homeName + ' Scorers';
        document.getElementById('player-away-title').innerText = awayName + ' Scorers';

        // Badge Confidence & Timestamp
        const badge = document.getElementById('confidence-badge');
        badge.className = 'badge';
        badge.innerText = `${anal.confidence} Confidence`;
        badge.classList.add(`badge-${anal.confidence.toLowerCase().replace(' ', '-')}`);

        document.getElementById('match-timestamp').innerText = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Scoreboard
        document.getElementById('score-home-name').innerText = homeName;
        document.getElementById('score-away-name').innerText = awayName;
        document.getElementById('score-val').innerText = `${homeGoals} - ${awayGoals}`;
        document.getElementById('xg-home').innerText = `xG ${pred.player_prediction.home_team.clean_sheet_prediction.probability ? (anal.home_stats.recent_form_gf).toFixed(2) : '0.00'}`;
        
        // Wait, let's look at expected goals in analytics instead of placeholders
        const home_xg = anal.home_stats.recent_form_gf;
        const away_xg = anal.away_stats.recent_form_gf;
        document.getElementById('xg-home').innerText = `Form xG: ${home_xg}`;
        document.getElementById('xg-away').innerText = `Form xG: ${away_xg}`;

        // Winner Text
        const winnerTxt = document.getElementById('predicted-winner-text');
        if (anal.winner === 'Draw') {
            winnerTxt.innerText = 'Predicted Draw';
        } else {
            winnerTxt.innerText = `${anal.winner} Win`;
        }

        // Win Probabilities
        const pHome = pred.match_prediction.win_probabilities.home_team.probability;
        const pDraw = pred.match_prediction.win_probabilities.draw.probability;
        const pAway = pred.match_prediction.win_probabilities.away_team.probability;

        document.getElementById('prob-home-label').innerText = `${homeName} Win`;
        document.getElementById('prob-away-label').innerText = `${awayName} Win`;

        document.getElementById('prob-home-val').innerText = `${pHome}%`;
        document.getElementById('prob-draw-val').innerText = `${pDraw}%`;
        document.getElementById('prob-away-val').innerText = `${pAway}%`;

        document.getElementById('prob-home-bar').style.width = `${pHome}%`;
        document.getElementById('prob-draw-bar').style.width = `${pDraw}%`;
        document.getElementById('prob-away-bar').style.width = `${pAway}%`;

        // Penalty Shootout Forecast Panel
        const shootoutCard = document.getElementById('shootout-card');
        if (shootoutCard) {
            if (pred.penalty_shootout && pred.penalty_shootout.show_shootout) {
                shootoutCard.classList.remove('hidden');
                document.getElementById('shootout-predicted-winner').innerText = pred.penalty_shootout.predicted_winner;
                
                document.getElementById('shootout-home-label').innerText = homeName;
                document.getElementById('shootout-home-bar').style.width = `${pred.penalty_shootout.home_win_probability}%`;
                document.getElementById('shootout-home-val').innerText = `${pred.penalty_shootout.home_win_probability}%`;
                
                document.getElementById('shootout-away-label').innerText = awayName;
                document.getElementById('shootout-away-bar').style.width = `${pred.penalty_shootout.away_win_probability}%`;
                document.getElementById('shootout-away-val').innerText = `${pred.penalty_shootout.away_win_probability}%`;
            } else {
                shootoutCard.classList.add('hidden');
            }
        }

        // Goal Insights
        const fts = pred.goal_insights.first_team_to_score;
        document.getElementById('insight-first-team').innerText = `${fts.team} (${fts.probability}%)`;

        const btts = pred.goal_insights.both_teams_to_score;
        document.getElementById('insight-btts').innerText = `${btts.prediction ? 'Yes' : 'No'} (${btts.probability}%)`;

        // Why Explanations
        const explList = document.getElementById('explanation-list');
        explList.innerHTML = '';
        anal.explanations.forEach(exp => {
            const li = document.createElement('li');
            li.innerText = exp;
            explList.appendChild(li);
        });

        // Goalkeepers
        const gkHome = pred.player_prediction.home_team.clean_sheet_prediction;
        const gkAway = pred.player_prediction.away_team.clean_sheet_prediction;

        document.getElementById('gk-home-name').innerText = gkHome.goalkeeper || 'N/A';
        document.getElementById('gk-away-name').innerText = gkAway.goalkeeper || 'N/A';
        
        document.getElementById('gk-home-cs').innerText = `Clean Sheet: ${gkHome.probability}%`;
        document.getElementById('gk-away-cs').innerText = `Clean Sheet: ${gkAway.probability}%`;

        // Player Scorers
        const homeScorersList = document.getElementById('player-home-scorers');
        homeScorersList.innerHTML = '';
        if (pred.player_prediction.home_team.goal.length === 0) {
            homeScorersList.innerHTML = '<li class="text-center text-muted">No goal predicted</li>';
        } else {
            pred.player_prediction.home_team.goal.forEach(p => {
                const li = document.createElement('li');
                li.innerHTML = `<span class="name">${p.name}</span><span class="prob">${p.predictions[0].probability}%</span>`;
                homeScorersList.appendChild(li);
            });
        }

        const awayScorersList = document.getElementById('player-away-scorers');
        awayScorersList.innerHTML = '';
        if (pred.player_prediction.away_team.goal.length === 0) {
            awayScorersList.innerHTML = '<li class="text-center text-muted">No goal predicted</li>';
        } else {
            pred.player_prediction.away_team.goal.forEach(p => {
                const li = document.createElement('li');
                li.innerHTML = `<span class="name">${p.name}</span><span class="prob">${p.predictions[0].probability}%</span>`;
                awayScorersList.appendChild(li);
            });
        }

        // Manager Comparisons
        document.getElementById('mgr-home-name').innerText = anal.home_stats.manager;
        document.getElementById('mgr-home-rating').innerText = `Rating: ${anal.home_stats.manager_rating.toFixed(2)}`;
        document.getElementById('mgr-home-win').innerText = anal.home_stats.manager_win_pct;
        document.getElementById('mgr-home-loss').innerText = anal.home_stats.manager_loss_pct;
        document.getElementById('mgr-home-scored').innerText = anal.home_stats.manager_avg_scored;
        document.getElementById('mgr-home-conceded').innerText = anal.home_stats.manager_avg_conceded;

        document.getElementById('mgr-away-name').innerText = anal.away_stats.manager;
        document.getElementById('mgr-away-rating').innerText = `Rating: ${anal.away_stats.manager_rating.toFixed(2)}`;
        document.getElementById('mgr-away-win').innerText = anal.away_stats.manager_win_pct;
        document.getElementById('mgr-away-loss').innerText = anal.away_stats.manager_loss_pct;
        document.getElementById('mgr-away-scored').innerText = anal.away_stats.manager_avg_scored;
        document.getElementById('mgr-away-conceded').innerText = anal.away_stats.manager_avg_conceded;

        // Match Factors Table Body
        const factorsBody = document.getElementById('factors-table-body');
        factorsBody.innerHTML = '';

        const factorList = [
            { name: 'FIFA Ranking', home: anal.home_stats.fifa_rank, away: anal.away_stats.fifa_rank, lowerBetter: true },
            { name: 'Elo Rating', home: anal.home_stats.elo, away: anal.away_stats.elo },
            { name: 'Matches Played', home: anal.home_stats.matches, away: anal.away_stats.matches },
            { name: 'Goals Scored', home: anal.home_stats.goals_for, away: anal.away_stats.goals_against },
            { name: 'Goals Conceded', home: anal.home_stats.goals_against, away: anal.away_stats.goals_for, lowerBetter: true },
            { name: 'Recent Form (GF)', home: anal.home_stats.recent_form_gf, away: anal.away_stats.recent_form_gf },
            { name: 'Recent Form (GA)', home: anal.home_stats.recent_form_ga, away: anal.away_stats.recent_form_ga, lowerBetter: true },
            { name: 'WC 2026 Goals Scored', home: anal.home_stats.wc_goals_for, away: anal.away_stats.wc_goals_for },
            { name: 'WC 2026 Goals Conceded', home: anal.home_stats.wc_goals_against, away: anal.away_stats.wc_goals_against, lowerBetter: true }
        ];

        factorList.forEach(f => {
            const tr = document.createElement('tr');
            
            const homeVal = parseFloat(f.home) || 0;
            const awayVal = parseFloat(f.away) || 0;

            let homeClass = '';
            let awayClass = '';

            if (f.home !== 'N/A' && f.away !== 'N/A') {
                if (homeVal !== awayVal) {
                    const homeWins = f.lowerBetter ? (homeVal < awayVal) : (homeVal > awayVal);
                    if (homeWins) {
                        homeClass = 'outcome-home';
                    } else {
                        awayClass = 'outcome-away';
                    }
                }
            }

            tr.innerHTML = `
                <td class="${homeClass}">${f.home}</td>
                <td>${f.name}</td>
                <td class="${awayClass}">${f.away}</td>
            `;
            factorsBody.appendChild(tr);
        });

        // Show Dashboard
        predictionDashboard.classList.remove('hidden');
    }

    // 5. Prediction History
    function addToHistory(data) {
        const pred = data.prediction;
        const anal = data.analytics;

        const homeName = pred.score_prediction.predicted_scoreline.home_team;
        const awayName = pred.score_prediction.predicted_scoreline.away_team;
        const homeGoals = pred.score_prediction.predicted_scoreline.home_goals;
        const awayGoals = pred.score_prediction.predicted_scoreline.away_goals;

        const entry = {
            id: Date.now(),
            home_team: homeName,
            away_team: awayName,
            winner: anal.winner,
            score: `${homeGoals}-${awayGoals}`,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            fullData: data
        };

        predictionHistory.unshift(entry);
        if (predictionHistory.length > 10) {
            predictionHistory.pop();
        }

        try {
            sessionStorage.setItem('goalgpt_history', JSON.stringify(predictionHistory));
        } catch (e) {
            console.error(e);
        }

        renderHistory();
    }

    function renderHistory() {
        const rows = document.getElementById('history-rows');
        rows.innerHTML = '';

        if (predictionHistory.length === 0) {
            rows.innerHTML = `
                <tr id="empty-history">
                    <td colspan="6" class="text-center text-muted">No predictions simulated in this session.</td>
                </tr>
            `;
            return;
        }

        predictionHistory.forEach(entry => {
            const tr = document.createElement('tr');
            tr.style.cursor = 'pointer';
            
            let outcomeClass = 'outcome-draw';
            if (entry.winner === entry.home_team) outcomeClass = 'outcome-home';
            else if (entry.winner === entry.away_team) outcomeClass = 'outcome-away';

            tr.innerHTML = `
                <td><strong>${entry.home_team}</strong></td>
                <td><strong>${entry.away_team}</strong></td>
                <td><span class="outcome ${outcomeClass}">${entry.winner}</span></td>
                <td class="score">${entry.score}</td>
                <td>${entry.time}</td>
                <td class="action-cell">
                    <button class="delete-hist-btn" data-id="${entry.id}"><i class="fa-solid fa-trash-can"></i></button>
                </td>
            `;

            // Click row to reload simulation dashboard
            tr.addEventListener('click', (e) => {
                // If clicked trash button, skip loading
                if (e.target.closest('.delete-hist-btn')) return;
                renderDashboard(entry.fullData);
                window.scrollTo({ top: document.getElementById('prediction-dashboard').offsetTop - 100, behavior: 'smooth' });
            });

            // Trash button click
            tr.querySelector('.delete-hist-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                predictionHistory = predictionHistory.filter(h => h.id !== entry.id);
                try {
                    sessionStorage.setItem('goalgpt_history', JSON.stringify(predictionHistory));
                } catch (err) {
                    console.error(err);
                }
                renderHistory();
            });

            rows.appendChild(tr);
        });
    }

    // Clear history
    document.getElementById('clear-history').addEventListener('click', () => {
        predictionHistory = [];
        try {
            sessionStorage.removeItem('goalgpt_history');
        } catch (e) {
            console.error(e);
        }
        renderHistory();
    });

    // 6. Alert Handling
    function showError(title, message, suggestions = null) {
        errorTitle.innerText = title;
        errorDesc.innerText = message;
        errorPanel.classList.remove('hidden');
        
        if (suggestions && suggestions.length > 0) {
            errorSuggestions.innerHTML = '';
            suggestions.forEach(s => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.innerText = s;
                btn.type = 'button';
                btn.addEventListener('click', () => {
                    // Try to populate whichever field is incorrect
                    if (homeSearch.value === '' || allTeams.includes(homeSearch.value) === false) {
                        homeSearch.value = s;
                        selectedHome = s;
                    } else {
                        awaySearch.value = s;
                        selectedAway = s;
                    }
                    errorPanel.classList.add('hidden');
                });
                errorSuggestions.appendChild(btn);
            });
            errorSuggestions.classList.remove('hidden');
        } else {
            errorSuggestions.classList.add('hidden');
        }
    }
    }
});
