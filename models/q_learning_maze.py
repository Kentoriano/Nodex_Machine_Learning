import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import io
import base64

# ─── Maze Definition ──────────────────────────────────────────────────────────
# 0 = free cell, 1 = wall, 2 = goal
MAZE = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 2],
])

ROWS, COLS = MAZE.shape
N_STATES = ROWS * COLS           # 49 states
ACTIONS = [0, 1, 2, 3]          # 0=Up 1=Down 2=Left 3=Right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
ACTION_SYMBOLS = ["↑", "↓", "←", "→"]
START_STATE = 0                  # top-left (0,0)
GOAL_STATE = ROWS * COLS - 1    # bottom-right (6,6)

REWARD_GOAL   =  100
REWARD_WALL   =  -10
REWARD_STEP   =  -1

# ─── Environment helpers ──────────────────────────────────────────────────────
def state_to_pos(s):
    return s // COLS, s % COLS

def pos_to_state(r, c):
    return r * COLS + c

def step(state, action):
    r, c = state_to_pos(state)
    dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        return state, REWARD_WALL, False   # hit boundary
    if MAZE[nr, nc] == 1:
        return state, REWARD_WALL, False   # hit wall
    new_state = pos_to_state(nr, nc)
    if MAZE[nr, nc] == 2:
        return new_state, REWARD_GOAL, True
    return new_state, REWARD_STEP, False

# ─── Q-Learning training ──────────────────────────────────────────────────────
def train_q_learning(
    episodes=600,
    alpha=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.99,
    max_steps=300,
):
    Q = np.zeros((N_STATES, len(ACTIONS)))
    rewards_per_episode = []
    steps_per_episode   = []
    epsilon_per_episode = []
    success_per_episode = []
    epsilon = epsilon_start
    episode_summaries   = []

    for ep in range(episodes):
        state = START_STATE
        total_reward = 0
        done = False

        for step_n in range(max_steps):
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.choice(ACTIONS)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = step(state, action)
            # Q-Learning update
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step_n + 1)
        epsilon_per_episode.append(epsilon)
        success_per_episode.append(int(done))

        if ep % 50 == 0 or ep == episodes - 1:
            episode_summaries.append({
                "episode": ep + 1,
                "total_reward": round(float(total_reward), 2),
                "steps": step_n + 1,
                "success": done,
                "epsilon": round(epsilon, 4),
            })

    # Derive optimal path using greedy policy
    optimal_path = _greedy_path(Q)

    return {
        "Q": Q,
        "rewards": rewards_per_episode,
        "steps": steps_per_episode,
        "epsilons": epsilon_per_episode,
        "successes": success_per_episode,
        "summaries": episode_summaries,
        "optimal_path": optimal_path,
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
    }

def _greedy_path(Q, max_steps=200):
    state = START_STATE
    path  = [state]
    visited = set()
    for _ in range(max_steps):
        if state == GOAL_STATE:
            break
        if state in visited:
            break
        visited.add(state)
        action = int(np.argmax(Q[state]))
        next_state, _, done = step(state, action)
        state = next_state
        path.append(state)
        if done:
            break
    return path

# ─── Graph generators ─────────────────────────────────────────────────────────
def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def plot_reward_progression(results):
    rewards = results["rewards"]
    window  = 20
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(rewards, color="#30363d", linewidth=0.7, label="Raw reward")
    ax.plot(range(window-1, len(rewards)), smoothed,
            color="#58a6ff", linewidth=2, label=f"Moving avg ({window} ep)")
    ax.set_title("Reward Progression per Episode", color="white", fontsize=13)
    ax.set_xlabel("Episode", color="#8b949e")
    ax.set_ylabel("Total Reward", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="white", edgecolor="#30363d")
    return _fig_to_b64(fig)

def plot_success_rate(results):
    successes = results["successes"]
    window    = 50
    rate = np.convolve(successes, np.ones(window)/window, mode='valid') * 100

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(range(window-1, len(successes)), rate,
            color="#3fb950", linewidth=2)
    ax.fill_between(range(window-1, len(successes)), rate,
                    alpha=0.15, color="#3fb950")
    ax.set_title("Success Rate Evolution (rolling 50 episodes)", color="white", fontsize=13)
    ax.set_xlabel("Episode", color="#8b949e")
    ax.set_ylabel("Success Rate (%)", color="#8b949e")
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    return _fig_to_b64(fig)

def plot_epsilon_decay(results):
    epsilons = results["epsilons"]
    fig, ax  = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(epsilons, color="#f78166", linewidth=2)
    ax.set_title("Epsilon Decay (Exploration → Exploitation)", color="white", fontsize=13)
    ax.set_xlabel("Episode", color="#8b949e")
    ax.set_ylabel("Epsilon (ε)", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    return _fig_to_b64(fig)

def plot_maze_path(results):
    Q    = results["Q"]
    path = results["optimal_path"]

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    cell_colors = {0: "#161b22", 1: "#30363d", 2: "#238636"}
    for r in range(ROWS):
        for c in range(COLS):
            color = cell_colors[MAZE[r, c]]
            ax.add_patch(mpatches.Rectangle(
                (c, ROWS - 1 - r), 1, 1,
                facecolor=color, edgecolor="#484f58", linewidth=0.8
            ))

    # Draw optimal path
    path_coords = [state_to_pos(s) for s in path]
    for i in range(len(path_coords) - 1):
        r1, c1 = path_coords[i]
        r2, c2 = path_coords[i + 1]
        ax.annotate(
            "", xy=(c2 + 0.5, ROWS - 1 - r2 + 0.5),
            xytext=(c1 + 0.5, ROWS - 1 - r1 + 0.5),
            arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=2)
        )

    # Start / Goal markers
    sr, sc = state_to_pos(START_STATE)
    gr, gc = state_to_pos(GOAL_STATE)
    ax.text(sc + 0.5, ROWS - 1 - sr + 0.5, "S",
            ha="center", va="center", fontsize=14, color="#e3b341", fontweight="bold")
    ax.text(gc + 0.5, ROWS - 1 - gr + 0.5, "G",
            ha="center", va="center", fontsize=14, color="white", fontweight="bold")

    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks(range(COLS + 1))
    ax.set_yticks(range(ROWS + 1))
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.set_title("Optimal Path Learned by Agent", color="white", fontsize=13)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    legend_elements = [
        mpatches.Patch(facecolor="#161b22", edgecolor="#484f58", label="Free cell"),
        mpatches.Patch(facecolor="#30363d", edgecolor="#484f58", label="Wall"),
        mpatches.Patch(facecolor="#238636", edgecolor="#484f58", label="Goal"),
    ]
    ax.legend(handles=legend_elements, facecolor="#161b22",
              labelcolor="white", edgecolor="#30363d", loc="upper right", fontsize=8)
    return _fig_to_b64(fig)

def plot_q_heatmap(results):
    Q = results["Q"]
    best_q = np.max(Q, axis=1).reshape(ROWS, COLS)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    # Mask walls
    masked = np.ma.masked_where(MAZE == 1, best_q)
    im = ax.imshow(masked, cmap="Blues", interpolation="nearest")

    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r, c] == 1:
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#30363d", edgecolor="#484f58"
                ))
            else:
                action = int(np.argmax(Q[pos_to_state(r, c)]))
                ax.text(c, r, ACTION_SYMBOLS[action],
                        ha="center", va="center", fontsize=14, color="white")

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
    cbar.set_label("Max Q-value", color="#8b949e")

    ax.set_title("Q-value Heatmap + Best Action per State", color="white", fontsize=13)
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    return _fig_to_b64(fig)

def plot_steps_per_episode(results):
    steps   = results["steps"]
    window  = 20
    smoothed = np.convolve(steps, np.ones(window)/window, mode='valid')

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(steps, color="#30363d", linewidth=0.7, label="Raw steps")
    ax.plot(range(window-1, len(steps)), smoothed,
            color="#d2a8ff", linewidth=2, label=f"Moving avg ({window} ep)")
    ax.set_title("Steps per Episode (Learning Efficiency)", color="white", fontsize=13)
    ax.set_xlabel("Episode", color="#8b949e")
    ax.set_ylabel("Steps to Finish", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="white", edgecolor="#30363d")
    return _fig_to_b64(fig)

# ─── Full results bundle ───────────────────────────────────────────────────────
def get_rl_results():
    results = train_q_learning()
    Q       = results["Q"]

    # Metrics
    rewards      = results["rewards"]
    successes    = results["successes"]
    last_100     = rewards[-100:]
    final_sr     = sum(successes[-100:])
    path_len     = len(results["optimal_path"]) - 1
    best_reward  = round(float(max(rewards)), 2)
    avg_reward   = round(float(np.mean(last_100)), 2)

    # Q-table sample (first 10 non-wall states)
    q_table_sample = []
    for s in range(N_STATES):
        r, c = state_to_pos(s)
        if MAZE[r, c] != 1:
            row = {"state": s, "pos": f"({r},{c})"}
            for i, a in enumerate(ACTION_NAMES):
                row[a] = round(float(Q[s, i]), 3)
            row["best_action"] = ACTION_NAMES[int(np.argmax(Q[s]))]
            q_table_sample.append(row)
        if len(q_table_sample) >= 12:
            break

    # Graphs
    graphs = {
        "reward_progression": plot_reward_progression(results),
        "success_rate":       plot_success_rate(results),
        "epsilon_decay":      plot_epsilon_decay(results),
        "maze_path":          plot_maze_path(results),
        "q_heatmap":          plot_q_heatmap(results),
        "steps_per_episode":  plot_steps_per_episode(results),
    }

    return {
        "metrics": {
            "n_states":    N_STATES,
            "n_actions":   len(ACTIONS),
            "episodes":    results["episodes"],
            "alpha":       results["alpha"],
            "gamma":       results["gamma"],
            "best_reward": best_reward,
            "avg_reward_last100": avg_reward,
            "success_rate_last100": final_sr,
            "optimal_path_length": path_len,
            "reached_goal": results["optimal_path"][-1] == GOAL_STATE,
        },
        "summaries": results["summaries"],
        "q_table_sample": q_table_sample,
        "graphs": graphs,
        "maze": MAZE.tolist(),
        "optimal_path": results["optimal_path"],
    }
