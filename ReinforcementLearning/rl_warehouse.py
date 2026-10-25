"""
rl_warehouse.py
===============
Tabular reinforcement learning on an 11x11 warehouse grid.

The module contains
  * the warehouse MDP (optionally stochastic, via a "slip" probability),
  * an exact solver (value iteration) and a BFS shortest-path oracle,
    used as ground truth to *measure* how good a learned policy is,
  * three learners: Q-learning, SARSA and REINFORCE (with a baseline),
  * evaluation helpers and plotting utilities.

Conventions (used consistently in code, README and report)
  * epsilon is the probability of taking a RANDOM action (exploration).
  * Reward r_{t+1} is received on entering the next state s_{t+1}.
  * Terminal states (goal and crash squares) have value 0 and are never
    bootstrapped from.
  * All randomness comes from numpy Generators seeded by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
ACTIONS = ("up", "right", "down", "left")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
N_ROWS = N_COLS = 11
N_STATES = N_ROWS * N_COLS
N_ACTIONS = 4
GOAL = (0, 5)  # packaging area

# Free squares ("aisles") of the warehouse, row -> columns.
AISLES = {
    1: list(range(1, 10)),
    2: [1, 7, 9],
    3: list(range(1, 8)) + [9],
    4: [3, 7],
    5: list(range(11)),
    6: [5],
    7: list(range(1, 10)),
    8: [3, 7],
    9: list(range(11)),
}


def to_state(r: int, c: int) -> int:
    return r * N_COLS + c


def to_rc(s: int) -> tuple[int, int]:
    return divmod(s, N_COLS)


class WarehouseEnv:
    """Grid warehouse. Reaching GOAL gives +100, free squares cost -1 per step,
    and stepping on a shelf (any other square) gives -100 and ends the episode.

    Parameters
    ----------
    slip : probability that the executed action is replaced by a uniformly
           random one (0 = deterministic, as in the original project).
    """

    def __init__(self, slip: float = 0.0, goal_reward: float = 100.0,
                 step_reward: float = -1.0, crash_reward: float = -100.0):
        self.slip = float(slip)
        self.goal_reward, self.step_reward, self.crash_reward = (
            goal_reward, step_reward, crash_reward)

        # reward received on ENTERING each state
        self.R = np.full(N_STATES, crash_reward)
        for r, cols in AISLES.items():
            for c in cols:
                self.R[to_state(r, c)] = step_reward
        self.goal = to_state(*GOAL)
        self.R[self.goal] = goal_reward
        self.terminal = self.R != step_reward  # goal + shelves

        # deterministic transition table NS[s, a] (walls: stay in place)
        self.NS = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
        for s in range(N_STATES):
            r, c = to_rc(s)
            for a, (dr, dc) in enumerate(DELTAS):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < N_ROWS and 0 <= c2 < N_COLS:
                    self.NS[s, a] = to_state(r2, c2)
                else:
                    self.NS[s, a] = s
        self.free_states = np.flatnonzero(~self.terminal)  # 56 squares

    # -- interaction -------------------------------------------------------- #
    def step(self, s: int, a: int, rng: np.random.Generator):
        if self.slip > 0.0 and rng.random() < self.slip:
            a = int(rng.random() * N_ACTIONS)
        s2 = int(self.NS[s, a])
        return s2, float(self.R[s2]), bool(self.terminal[s2])

    def random_start(self, rng: np.random.Generator) -> int:
        return int(self.free_states[int(rng.random() * len(self.free_states))])

    def with_slip(self, slip: float) -> "WarehouseEnv":
        return WarehouseEnv(slip, self.goal_reward, self.step_reward,
                            self.crash_reward)


# --------------------------------------------------------------------------- #
# Ground truth: value iteration + BFS
# --------------------------------------------------------------------------- #
def value_iteration(env: WarehouseEnv, gamma: float = 0.9, tol: float = 1e-10,
                    max_iter: int = 10_000):
    """Solve the Bellman optimality equations exactly (dynamic programming).

        Q*(s,a) = sum_{s'} P(s'|s,a) [ R(s') + gamma * V*(s') ],  V*(s)=max_a Q*(s,a)

    with V* = 0 on terminal states.  Returns (Q*, V*, greedy policy, #iterations).
    """
    V = np.zeros(N_STATES)
    for it in range(1, max_iter + 1):
        # value of *attempting* each action deterministically
        X = env.R[env.NS] + gamma * V[env.NS] * (~env.terminal[env.NS])
        Q = (1 - env.slip) * X + env.slip * X.mean(axis=1, keepdims=True)
        V_new = np.where(env.terminal, 0.0, Q.max(axis=1))
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = np.where(env.terminal[:, None], 0.0, Q)
    return Q, V, Q.argmax(axis=1), it


def epsilon_soft_optimal(env: WarehouseEnv, gamma: float = 0.9, eps: float = 0.1,
                         tol: float = 1e-10, max_iter: int = 20_000) -> np.ndarray:
    """Exact fixed point of SARSA: the optimal action values *among
    epsilon-greedy policies*,

        Q_eps(s,a) = E[ r + gamma * ( (1-eps) max_a' Q_eps(s',a') + eps mean_a' Q_eps(s',a') ) ].

    SARSA converges to this table (not to Q*) when epsilon is kept constant.
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    for _ in range(max_iter):
        Vn = np.where(env.terminal, 0.0, (1 - eps) * Q.max(1) + eps * Q.mean(1))
        X = env.R[env.NS] + gamma * Vn[env.NS]
        Qn = (1 - env.slip) * X + env.slip * X.mean(axis=1, keepdims=True)
        Qn = np.where(env.terminal[:, None], 0.0, Qn)
        done = np.abs(Qn - Q).max() < tol
        Q = Qn
        if done:
            break
    return Q


def shortest_lengths(env: WarehouseEnv) -> np.ndarray:
    """BFS distance (in steps) from every free square to the goal while avoiding
    shelves.  Used as an oracle independent of any value function."""
    dist = np.full(N_STATES, np.inf)
    dist[env.goal] = 0
    changed = True
    while changed:
        changed = False
        for s in env.free_states:
            for a in range(N_ACTIONS):
                s2 = env.NS[s, a]
                if (s2 == env.goal or not env.terminal[s2]) and dist[s2] + 1 < dist[s]:
                    dist[s] = dist[s2] + 1
                    changed = True
    return dist


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def greedy_action(q_row: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """argmax with uniformly random tie-breaking (np.argmax would always pick
    'up' on ties, which biases exploration early in training)."""
    if rng is None:
        return int(np.argmax(q_row))
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0]) if len(best) == 1 else int(best[int(rng.random() * len(best))])


def epsilon_greedy(q_row: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.random() * N_ACTIONS)
    return greedy_action(q_row, rng)


Schedule = Union[float, Callable[[int], float]]


def linear_decay(start: float, end: float, n_episodes: int) -> Callable[[int], float]:
    """epsilon_k decreases linearly from `start` to `end` over n_episodes."""
    return lambda k: max(end, start + (end - start) * k / max(1, n_episodes - 1))


def _value(sched: Schedule, k: int) -> float:
    return sched(k) if callable(sched) else sched


# --------------------------------------------------------------------------- #
# Evaluation of a greedy policy
# --------------------------------------------------------------------------- #
def greedy_path(env: WarehouseEnv, scores: np.ndarray, start: int,
                max_steps: int = 60):
    """Follow argmax(scores) from `start` in the DETERMINISTIC dynamics.
    Returns (path, outcome) with outcome in {'goal','crash','loop'}."""
    s, path = start, [start]
    for _ in range(max_steps):
        if env.terminal[s]:
            break
        s = int(env.NS[s, int(np.argmax(scores[s]))])
        path.append(s)
    if s == env.goal:
        return path, "goal"
    return path, ("crash" if env.terminal[s] else "loop")


def evaluate_greedy(env: WarehouseEnv, scores: np.ndarray, dist: np.ndarray):
    """Exact evaluation of the greedy policy from every free square
    (deterministic dynamics).  Returns
        success    : fraction of starts that reach the goal,
        optimal    : fraction of starts whose path is a shortest path,
        excess     : mean extra steps over the shortest path (successful starts)."""
    ok, opt, excess = 0, 0, []
    for s0 in env.free_states:
        path, out = greedy_path(env, scores, int(s0))
        if out == "goal":
            ok += 1
            L = len(path) - 1
            opt += int(L == dist[s0])
            excess.append(L - dist[s0])
    n = len(env.free_states)
    return ok / n, opt / n, (float(np.mean(excess)) if excess else float("nan"))


def monte_carlo_eval(env: WarehouseEnv, scores: np.ndarray, n_episodes: int,
                     rng: np.random.Generator, max_steps: int = 200):
    """Run the greedy policy in `env` (possibly stochastic) from random starts.
    Returns dict(return, goal, crash, timeout) averaged over episodes."""
    ret, goal, crash = [], 0, 0
    for _ in range(n_episodes):
        s = env.random_start(rng)
        G, done = 0.0, False
        for _ in range(max_steps):
            s, r, done = env.step(s, int(np.argmax(scores[s])), rng)
            G += r
            if done:
                break
        ret.append(G)
        if done:
            if s == env.goal:
                goal += 1
            else:
                crash += 1
    return dict(ret=float(np.mean(ret)), goal=goal / n_episodes,
                crash=crash / n_episodes,
                timeout=1 - (goal + crash) / n_episodes)


# --------------------------------------------------------------------------- #
# Learners
# --------------------------------------------------------------------------- #
@dataclass
class History:
    """Per-episode training record + periodic greedy evaluation."""
    returns: np.ndarray        # undiscounted return of each training episode
    steps: np.ndarray          # episode length
    outcome: np.ndarray        # 0 = timeout, 1 = goal, 2 = crash
    eval_episodes: np.ndarray  # episodes at which the greedy policy was evaluated
    eval_success: np.ndarray
    eval_optimal: np.ndarray
    eval_rmse: np.ndarray      # RMSE of V(s)=max_a Q(s,a) to that of `q_target` (NaN if none)


def _make_history(n, eval_every):
    ev = np.arange(0, n + 1, eval_every) if eval_every else np.array([], dtype=int)
    return (np.zeros(n), np.zeros(n, dtype=int), np.zeros(n, dtype=int), ev,
            np.full(len(ev), np.nan), np.full(len(ev), np.nan),
            np.full(len(ev), np.nan))


def train_td(env: WarehouseEnv, algo: str = "q", episodes: int = 1000,
             alpha: float = 0.5, gamma: float = 0.9, epsilon: Schedule = 0.1,
             seed: int = 0, max_steps: int = 200, eval_every: int = 0,
             dist: np.ndarray | None = None, q_target: np.ndarray | None = None):
    """Tabular Q-learning (off-policy) or SARSA (on-policy).

    Q-learning : Q(s,a) += alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]
    SARSA      : Q(s,a) += alpha [ r + gamma Q(s',a')        - Q(s,a) ]  (a' ~ eps-greedy)

    The bootstrap term is 0 when s' is terminal.
    """
    assert algo in ("q", "sarsa")
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    ret, steps, outc, ev, ev_s, ev_o, ev_r = _make_history(episodes, eval_every)

    def snapshot(k):
        if eval_every and k % eval_every == 0:
            i = k // eval_every
            ev_s[i], ev_o[i], _ = evaluate_greedy(env, Q, dist)
            if q_target is not None:
                diff = (Q.max(1) - q_target.max(1))[env.free_states]
                ev_r[i] = float(np.sqrt(np.mean(diff ** 2)))

    for k in range(episodes):
        snapshot(k)
        eps = _value(epsilon, k)
        s = env.random_start(rng)
        a = epsilon_greedy(Q[s], eps, rng)
        G, n, done = 0.0, 0, False
        while not done and n < max_steps:
            s2, r, done = env.step(s, a, rng)
            if algo == "sarsa" and not done:
                a2 = epsilon_greedy(Q[s2], eps, rng)  # on-policy: a' drawn BEFORE the update
            if done:
                target = r
            elif algo == "q":
                target = r + gamma * Q[s2].max()
            else:
                target = r + gamma * Q[s2, a2]
            Q[s, a] += alpha * (target - Q[s, a])
            if algo == "q" and not done:
                a2 = epsilon_greedy(Q[s2], eps, rng)  # off-policy: act AFTER the update
            if not done:
                a = a2
            s = s2
            G += r
            n += 1
        ret[k], steps[k] = G, n
        outc[k] = (1 if s == env.goal else 2) if done else 0
    snapshot(episodes)
    return Q, History(ret, steps, outc, ev, ev_s, ev_o, ev_r)


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def train_reinforce(env: WarehouseEnv, episodes: int = 3000, alpha: float = 0.01,
                    gamma: float = 0.9, baseline: bool = True, seed: int = 0,
                    max_steps: int = 200, eval_every: int = 0,
                    dist: np.ndarray | None = None):
    """Tabular REINFORCE with a softmax policy pi_theta(a|s) = softmax(theta[s]).

        theta[s] += alpha * (G_t - b(s)) * grad log pi(a_t|s_t)
        grad_{theta[s]} log pi(a|s) = onehot(a) - pi(.|s)

    b(s) is a running mean of the returns observed from s (variance reduction;
    it does not bias the gradient because it does not depend on the action).
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros((N_STATES, N_ACTIONS))
    b, cnt = np.zeros(N_STATES), np.zeros(N_STATES)
    ret, steps, outc, ev, ev_s, ev_o, ev_r = _make_history(episodes, eval_every)

    def snapshot(k):
        if eval_every and k % eval_every == 0:
            i = k // eval_every
            ev_s[i], ev_o[i], _ = evaluate_greedy(env, theta, dist)

    for k in range(episodes):
        snapshot(k)
        s = env.random_start(rng)
        traj, done, n = [], False, 0
        while not done and n < max_steps:
            a = int(np.searchsorted(np.cumsum(softmax(theta[s])), rng.random()))
            a = min(a, N_ACTIONS - 1)
            s2, r, done = env.step(s, a, rng)
            traj.append((s, a, r))
            s, n = s2, n + 1
        G = 0.0
        for (st, at, rt) in reversed(traj):
            G = rt + gamma * G
            adv = G - (b[st] if baseline else 0.0)
            pi = softmax(theta[st])
            grad = -pi
            grad[at] += 1.0
            theta[st] += alpha * adv * grad
            cnt[st] += 1
            b[st] += (G - b[st]) / cnt[st]
        ret[k], steps[k] = sum(t[2] for t in traj), n
        outc[k] = (1 if s == env.goal else 2) if done else 0
    snapshot(episodes)
    return theta, History(ret, steps, outc, ev, ev_s, ev_o, ev_r)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def smooth(x: np.ndarray, w: int = 25) -> np.ndarray:
    """Trailing moving average (same length as x)."""
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def mean_ci(a: np.ndarray, axis: int = 0, z: float = 1.96):
    """Mean and normal-approximation 95% half-width across seeds."""
    m = a.mean(axis=axis)
    h = z * a.std(axis=axis, ddof=1) / np.sqrt(a.shape[axis])
    return m, h


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def draw_grid(ax, env: WarehouseEnv, paths=(), policy=None, values=None,
              title=None, cmap="viridis", annotate=False):
    """Draw the warehouse. Shelves are dark, aisles white, goal green.
    Optionally overlay paths (list of state lists), a policy (arrows) or a
    value heat-map."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = np.zeros((N_ROWS, N_COLS))
    if values is not None:
        img = np.where(env.terminal, np.nan, values).reshape(N_ROWS, N_COLS)
        im = ax.imshow(img, cmap=cmap, origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for s in range(N_STATES):
        r, c = to_rc(s)
        if env.terminal[s] and s != env.goal:
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#3b3f4a"))
        elif s == env.goal:
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#2e9e5b"))
            ax.text(c, r, "G", ha="center", va="center", color="white",
                    fontweight="bold")
        elif values is None:
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#f4f1ea"))
            if annotate:
                ax.text(c, r, f"{r},{c}", ha="center", va="center", fontsize=5,
                        color="#888")
    if policy is not None:
        arrows = {0: (0, -.32), 1: (.32, 0), 2: (0, .32), 3: (-.32, 0)}
        for s in env.free_states:
            r, c = to_rc(int(s))
            dx, dy = arrows[int(policy[s])]
            ax.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=.17,
                     head_length=.12, fc="#b5483a", ec="#b5483a", length_includes_head=True)
    colors = ["#1f5fa8", "#d98324", "#8e44ad", "#c0392b"]
    for i, p in enumerate(paths):
        rr = [to_rc(s)[0] for s in p]
        cc = [to_rc(s)[1] for s in p]
        ax.plot(cc, rr, "-o", color=colors[i % 4], lw=2, ms=4, alpha=.9)
        ax.plot(cc[0], rr[0], "s", color=colors[i % 4], ms=8)
    ax.set_xlim(-.5, N_COLS - .5)
    ax.set_ylim(N_ROWS - .5, -.5)
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-.5, N_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N_ROWS, 1), minor=True)
    ax.grid(which="minor", color="#cfcabf", lw=.5)
    ax.tick_params(which="minor", length=0)
    if title:
        ax.set_title(title, fontsize=10)
