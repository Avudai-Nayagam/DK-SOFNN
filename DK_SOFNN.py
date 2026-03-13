"""
DK-SOFNN: Data-Knowledge-Driven Self-Organizing Fuzzy Neural Network
=====================================================================
Implementation based on:
    H. Han, X. Liu, and J. Qiao,
    "Data-Knowledge-Driven Self-Organizing Fuzzy Neural Network,"
    IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 2, Feb. 2024.

This file provides a faithful, standalone NumPy implementation of the
full DK-SOFNN algorithm as described in Table II and Eqs. (1)-(30),
(42)-(44) of the paper.

Usage:
    python DK_SOFNN.py
"""

import numpy as np
import copy
import time

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    # Minimal fallback if sklearn is not installed
    class MinMaxScaler:
        def fit_transform(self, X):
            self.min_ = X.min(axis=0)
            self.max_ = X.max(axis=0)
            self.scale_ = self.max_ - self.min_
            self.scale_[self.scale_ == 0] = 1.0
            return (X - self.min_) / self.scale_
        def transform(self, X):
            return (X - self.min_) / self.scale_
        def inverse_transform(self, X):
            return X * self.scale_ + self.min_

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS (Paper experimental parameters)
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
INITIAL_RULES = 20          # Paper: 20 initial rules
SOURCE_EPOCHS = 300          # Paper: I = 300 epochs for source
SOURCE_LR = 0.1              # Paper: λ_S = 0.1
TARGET_LR = 0.01             # Paper: λ_T = 0.01
N_T_DEFAULT = 50             # Paper: N_T = 50 for target
MIN_RULES = 3                # Lower bound on rule count
MAX_RULES = 30               # Upper bound on rule count
MIN_WIDTH = 0.01             # Minimum width to avoid division by zero
EPS = 1e-8                   # Numerical stability constant

# ═══════════════════════════════════════════════════════════════════════════════
#  Five-Layer Fuzzy Neural Network (Eqs. 1-5)
# ═══════════════════════════════════════════════════════════════════════════════


class FNN:
    """Five-layer Fuzzy Neural Network.

    Architecture (Paper Eqs. 1-5):
        Layer 1 — Input layer:        x_p(t),  p = 1,...,P
        Layer 2 — Membership (RBF):   φ_{k,p} = exp(-(x_p - c_{k,p})^2 / (2σ_{k,p}^2))  [Eq. 2]
        Layer 3 — Rule layer:         u_k = Π_p φ_{k,p}                                   [Eq. 3]
        Layer 4 — Normalized layer:   v_k = u_k / Σ_j u_j                                 [Eq. 4]
        Layer 5 — Output layer:       y   = Σ_k w_k * v_k                                 [Eq. 5]

    Parameters:
        n_inputs : int — number of input features (P)
        n_rules  : int — number of fuzzy rules (K)
    """

    def __init__(self, n_inputs, n_rules):
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        # Centers c_{k,p}, widths σ_{k,p}, consequent weights w_k
        self.centers = np.random.rand(n_rules, n_inputs)
        self.widths = np.random.uniform(0.1, 0.4, (n_rules, n_inputs))
        self.weights = np.random.randn(n_rules, 1) * 0.1

    def compute_layers(self, x):
        """Forward pass through the 5-layer FNN for a single sample.

        Args:
            x: Input vector, shape (n_inputs,) or (1, n_inputs).

        Returns:
            y_pred : float — scalar network output
            v      : ndarray, shape (n_rules, 1) — normalized firing strengths
            u      : ndarray, shape (n_rules, 1) — raw firing strengths
        """
        x = x.reshape(1, -1)  # (1, P)

        # Layer 2: Gaussian membership — Eq. (2)
        # φ_{k,p} = exp(-(x_p - c_{k,p})^2 / (2 * σ_{k,p}^2))
        diff = x - self.centers                             # (K, P)
        safe_widths = np.clip(self.widths, MIN_WIDTH, None)
        phi = np.exp(-(diff ** 2) / (2.0 * safe_widths ** 2 + EPS))  # (K, P)

        # Layer 3: Rule firing strength — Eq. (3)
        # u_k = Π_p φ_{k,p}
        u = np.prod(phi, axis=1, keepdims=True)            # (K, 1)

        # Layer 4: Normalized firing strength — Eq. (4)
        # v_k = u_k / Σ_j u_j
        sum_u = np.sum(u) + EPS
        v = u / sum_u                                       # (K, 1)

        # Layer 5: Output — Eq. (5)
        # y = Σ_k w_k * v_k
        y_pred = float(np.dot(v.T, self.weights).item())     # scalar

        return y_pred, v, u

    def predict(self, X):
        """Batch prediction.

        Args:
            X: ndarray, shape (N, n_inputs).

        Returns:
            y_pred: ndarray, shape (N,).
        """
        preds = np.zeros(len(X))
        for i in range(len(X)):
            preds[i], _, _ = self.compute_layers(X[i])
        return preds

    def get_params(self):
        """Return a copy of (centers, widths, weights)."""
        return self.centers.copy(), self.widths.copy(), self.weights.copy()

    def clip_widths(self):
        """Ensure all widths stay above MIN_WIDTH for numerical stability."""
        self.widths = np.clip(self.widths, MIN_WIDTH, None)


# ═══════════════════════════════════════════════════════════════════════════════
#  Source FNN Training — Backpropagation (Eqs. 7-9)
# ═══════════════════════════════════════════════════════════════════════════════


def train_source_fnn(model, X_train, y_train, epochs=SOURCE_EPOCHS, lr=SOURCE_LR):
    """Train the source FNN using stochastic gradient descent.

    Paper Eqs. (7)-(9):
        Eq. (7):  Δw_k     = -λ_S * e * v_k
        Eq. (8):  Δc_{k,p} = -λ_S * e * (w_k - y) / Σu * u_k * (x_p - c_{k,p}) / σ_{k,p}^2
        Eq. (9):  Δσ_{k,p} = -λ_S * e * (w_k - y) / Σu * u_k * (x_p - c_{k,p})^2 / σ_{k,p}^3

    Args:
        model   : FNN instance (source)
        X_train : ndarray, shape (N, P) — normalized training features
        y_train : ndarray, shape (N, 1) — normalized training targets
        epochs  : int — number of training epochs (paper: I=300)
        lr      : float — learning rate λ_S (paper: 0.1)

    Returns:
        mse_history: list of per-epoch MSE values
    """
    mse_history = []
    n_samples = len(X_train)
    start = time.time()
    print(f"Training source FNN ({epochs} epochs, lr={lr}, {model.n_rules} rules)...")

    for epoch in range(epochs):
        epoch_error = 0.0
        indices = np.random.permutation(n_samples)

        for i in indices:
            x_sample = X_train[i]
            y_actual = y_train[i, 0]

            # Forward pass
            y_pred, v, u = model.compute_layers(x_sample)
            e = y_pred - y_actual
            epoch_error += e ** 2

            sum_u = np.sum(u) + EPS

            # Eq. (7): Weight gradient — ∂E/∂w_k = e * v_k
            grad_w = e * v                                       # (K, 1)

            # Common factor for center/width gradients
            # dy/du_k = (w_k - y_pred) / Σu
            dy_du = (model.weights - y_pred) / sum_u             # (K, 1)

            for k in range(model.n_rules):
                common = e * dy_du[k, 0] * u[k, 0]
                diff_k = x_sample - model.centers[k]
                safe_w = np.clip(model.widths[k], MIN_WIDTH, None)

                # Eq. (8): Center gradient
                grad_c = common * diff_k / (safe_w ** 2 + EPS)

                # Eq. (9): Width gradient
                grad_sigma = common * (diff_k ** 2) / (safe_w ** 3 + EPS)

                model.centers[k] -= lr * grad_c
                model.widths[k] -= lr * grad_sigma

            model.weights -= lr * grad_w
            model.clip_widths()

        mse = epoch_error / n_samples
        mse_history.append(mse)
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} — MSE: {mse:.6f}")

    elapsed = time.time() - start
    print(f"Source training complete in {elapsed:.1f}s  (final MSE: {mse_history[-1]:.6f})")
    return mse_history


# ═══════════════════════════════════════════════════════════════════════════════
#  Structure Knowledge Indexes (Eqs. 10-12)
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_indexes(model, X_data, y_data):
    """Compute the Similarity (R), Sensitivity (M), and Contribution (C) indexes.

    Paper Eqs. (10)-(12):
        Eq. (10) — Similarity R_l:
            R_l = 1/(K-1) * Σ_{k≠l} corr(u_l, u_k)
            Measures how correlated rule l is with every other rule.

        Eq. (11) — Sensitivity M_l:
            M_l = ū_l / Σ_k ū_k
            Ratio of rule l's average firing strength to the total.

        Eq. (12) — Contribution C_l:
            C_l = |corr(u_l, error)|
            Absolute correlation between rule l firing and prediction error.

    Args:
        model  : FNN instance
        X_data : ndarray, shape (N, P)
        y_data : ndarray, shape (N, 1) or (N,)

    Returns:
        R_indexes : ndarray, shape (K,)
        M_indexes : ndarray, shape (K,)
        C_indexes : ndarray, shape (K,)
    """
    n_samples = len(X_data)
    n_rules = model.n_rules
    y_flat = y_data.flatten()

    # Collect all rule activations and predictions
    all_u = np.zeros((n_samples, n_rules))
    all_y_pred = np.zeros(n_samples)

    for i in range(n_samples):
        y_p, _, u = model.compute_layers(X_data[i])
        all_u[i, :] = u.flatten()
        all_y_pred[i] = y_p

    # Average activation per rule
    u_bar = np.mean(all_u, axis=0)  # (K,)

    # ── Eq. (11): Sensitivity Index M_l ──
    # M_l = ū_l / Σ_k ū_k
    M_indexes = u_bar / (np.sum(u_bar) + EPS)

    # ── Eq. (10): Similarity Index R_l ──
    # R_l = 1/(K-1) * Σ_{k≠l} corr(u_l, u_k)
    R_indexes = np.zeros(n_rules)
    if n_rules > 1:
        # Center the activations
        u_centered = all_u - u_bar  # (N, K)
        # Norms for each rule
        norms = np.sqrt(np.sum(u_centered ** 2, axis=0) + EPS)  # (K,)
        for l in range(n_rules):
            corr_sum = 0.0
            for k in range(n_rules):
                if k == l:
                    continue
                cov_lk = np.sum(u_centered[:, l] * u_centered[:, k])
                corr_sum += cov_lk / (norms[l] * norms[k] + EPS)
            R_indexes[l] = corr_sum / (n_rules - 1)

    # ── Eq. (12): Contribution Index C_l ──
    # C_l = |corr(u_l, error)|
    errors = all_y_pred - y_flat
    err_centered = errors - np.mean(errors)
    err_norm = np.sqrt(np.sum(err_centered ** 2) + EPS)

    C_indexes = np.zeros(n_rules)
    u_centered = all_u - u_bar
    norms = np.sqrt(np.sum(u_centered ** 2, axis=0) + EPS)
    for l in range(n_rules):
        num = np.abs(np.sum(u_centered[:, l] * err_centered))
        C_indexes[l] = num / (norms[l] * err_norm + EPS)

    return R_indexes, M_indexes, C_indexes


# ═══════════════════════════════════════════════════════════════════════════════
#  Source Self-Organization (Eqs. 13-15)
# ═══════════════════════════════════════════════════════════════════════════════


def source_self_organization(model, X_data, y_data, n_iterations=5):
    """Iteratively grow/prune source FNN rules based on knowledge indexes.

    Paper Eqs. (13)-(15):
        Eq. (13) — Growing criterion:
            The rule l with min R, max M, max C is the most *effective*.
            Practically: M_l > M_avg AND C_l > C_avg → effective.
        Eq. (14) — New rule parameters:
            c_new = x_S(n)              (center at a training sample)
            σ_new = |x_S(n) - c_l| / √2 (width from distance to trigger rule)
            w_new = w_l                  (inherit trigger rule weight)
        Eq. (15) — Pruning criterion:
            The rule l with max R, min M, min C is the most *ineffective*.
            Practically: M_l < M_avg AND C_l < C_avg → ineffective.
            Delete the worst (lowest combined M + C).

    Args:
        model        : FNN instance (source)
        X_data       : ndarray, shape (N, P)
        y_data       : ndarray, shape (N, 1)
        n_iterations : int — number of grow/prune iterations

    Returns:
        model : modified FNN
    """
    print(f"Source self-organization ({n_iterations} iterations)...")
    print(f"  Initial rules: {model.n_rules}")

    for it in range(n_iterations):
        R_s, M_s, C_s = calculate_indexes(model, X_data, y_data)
        M_avg = np.mean(M_s)
        C_avg = np.mean(C_s)

        action = "None"

        # Eq. (13): Effective mask — candidates for growing
        effective = (M_s > M_avg) & (C_s > C_avg)
        # Eq. (15): Ineffective mask — candidates for pruning
        ineffective = (M_s < M_avg) & (C_s < C_avg)

        # Growing — Eq. (14)
        if np.any(effective) and model.n_rules < MAX_RULES:
            eff_idx = np.where(effective)[0]
            # Combined normalized score to find the best effective rule
            M_norm = M_s[eff_idx] / (np.max(M_s) + EPS)
            C_norm = C_s[eff_idx] / (np.max(C_s) + EPS)
            best_local = np.argmax(M_norm + C_norm)
            best_rule = eff_idx[best_local]

            # Eq. (14): New rule near the effective rule
            # Paper suggests c_new = x_S(n); we use best center + perturbation
            rand_idx = np.random.randint(len(X_data))
            new_center = X_data[rand_idx].reshape(1, -1)
            diff = np.abs(X_data[rand_idx] - model.centers[best_rule])
            new_width = np.clip(diff / np.sqrt(2.0), MIN_WIDTH, None).reshape(1, -1)
            new_weight = model.weights[best_rule].copy().reshape(1, -1)

            model.centers = np.vstack([model.centers, new_center])
            model.widths = np.vstack([model.widths, new_width])
            model.weights = np.vstack([model.weights, new_weight])
            model.n_rules += 1
            action = "Grow (Eq. 14)"

        # Pruning — Eq. (15)
        elif np.any(ineffective) and model.n_rules > MIN_RULES:
            ineff_idx = np.where(ineffective)[0]
            M_norm = M_s[ineff_idx] / (np.max(M_s) + EPS)
            C_norm = C_s[ineff_idx] / (np.max(C_s) + EPS)
            worst_local = np.argmin(M_norm + C_norm)
            worst_rule = ineff_idx[worst_local]

            model.centers = np.delete(model.centers, worst_rule, axis=0)
            model.widths = np.delete(model.widths, worst_rule, axis=0)
            model.weights = np.delete(model.weights, worst_rule, axis=0)
            model.n_rules -= 1
            action = "Prune (Eq. 15)"

        print(f"  Iter {it+1}: {action:20s} | Rules={model.n_rules} | "
              f"M_avg={M_avg:.4f} C_avg={C_avg:.4f}")

    print(f"  Final source rules: {model.n_rules}\n")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Target Structure Compensation (Eqs. 17-22)
# ═══════════════════════════════════════════════════════════════════════════════


def adjust_target_structure(target_model, source_avgs, source_expert,
                            X_target, y_target):
    """Adjust the target FNN structure by comparing target vs. source indexes.

    Paper Eqs. (17)-(22):

        Step 6 — Growing phase, Eqs. (17)-(18):
            IF R̄_T ≤ R̄_S AND M̄_T ≥ M̄_S AND C̄_T ≤ C̄_S → grow.
            New rule: c_new = x_T(t), σ_new = |x_T(t) - c_l| / 2,
                      w_new = w_l  (l = argmax C_T).

        Step 7 — Pruning phase, Eq. (19):
            IF (R̄_T ≥ R̄_S AND C̄_T ≥ C̄_S) OR (M̄_T ≤ M̄_S AND C̄_T ≥ C̄_S) → prune.
            Remove rule with lowest M_T.

        Step 8 — Compensating phase, Eqs. (20)-(21):
            IF (R̄_T ≥ R̄_S AND C̄_T ≤ C̄_S) OR (M̄_T ≤ M̄_S AND C̄_T ≤ C̄_S) → compensate.
            Replace worst target rule (min C_T) with source expert.

        Step 9 — Constant phase, Eq. (22):
            Otherwise, keep structure unchanged.

    Args:
        target_model  : FNN (target)
        source_avgs   : tuple (R̄_S, M̄_S, C̄_S)
        source_expert : dict with 'center', 'width', 'weight' of best source rule
        X_target      : ndarray, shape (N, P)
        y_target      : ndarray, shape (N, 1)

    Returns:
        action : str describing what was done
        n_rules: int — new rule count
        target_avgs: tuple (R̄_T, M̄_T, C̄_T)
    """
    R_s_avg, M_s_avg, C_s_avg = source_avgs

    # Compute current target indexes
    R_t, M_t, C_t = calculate_indexes(target_model, X_target, y_target)
    R_t_avg = np.mean(R_t)
    M_t_avg = np.mean(M_t)
    C_t_avg = np.mean(C_t)

    action = "Constant (Eq. 22)"  # Default: Step 9

    # Step 6: Growing — Eqs. (17)-(18)
    if R_t_avg <= R_s_avg and M_t_avg >= M_s_avg and C_t_avg <= C_s_avg:
        if target_model.n_rules < MAX_RULES:
            # Eq. (18): New rule at a target sample
            best_contrib = np.argmax(C_t)
            rand_idx = np.random.randint(len(X_target))
            new_center = X_target[rand_idx].reshape(1, -1)
            diff = np.abs(X_target[rand_idx] - target_model.centers[best_contrib])
            new_width = np.clip(diff / 2.0, MIN_WIDTH, None).reshape(1, -1)
            new_weight = target_model.weights[best_contrib].copy().reshape(1, -1)

            target_model.centers = np.vstack([target_model.centers, new_center])
            target_model.widths = np.vstack([target_model.widths, new_width])
            target_model.weights = np.vstack([target_model.weights, new_weight])
            target_model.n_rules += 1
            action = "Growing (Eqs. 17-18)"

    # Step 7: Pruning — Eq. (19)
    elif ((R_t_avg >= R_s_avg and C_t_avg >= C_s_avg) or
          (M_t_avg <= M_s_avg and C_t_avg >= C_s_avg)):
        if target_model.n_rules > MIN_RULES:
            worst_rule = np.argmin(M_t)
            target_model.centers = np.delete(target_model.centers, worst_rule, axis=0)
            target_model.widths = np.delete(target_model.widths, worst_rule, axis=0)
            target_model.weights = np.delete(target_model.weights, worst_rule, axis=0)
            target_model.n_rules -= 1
            action = "Pruning (Eq. 19)"

    # Step 8: Compensating — Eqs. (20)-(21)
    elif ((R_t_avg >= R_s_avg and C_t_avg <= C_s_avg) or
          (M_t_avg <= M_s_avg and C_t_avg <= C_s_avg)):
        worst_rule = np.argmin(C_t)
        target_model.centers[worst_rule] = source_expert['center'].copy()
        target_model.widths[worst_rule] = source_expert['width'].copy()
        target_model.weights[worst_rule] = source_expert['weight'].copy()
        action = "Compensating (Eqs. 20-21)"

    # Step 9: Constant — Eq. (22) — already set as default

    return action, target_model.n_rules, (R_t_avg, M_t_avg, C_t_avg)


# ═══════════════════════════════════════════════════════════════════════════════
#  Parameter Reinforcement Update (Eqs. 23-30)
# ═══════════════════════════════════════════════════════════════════════════════


def parameter_reinforcement_update(model, x_sample, y_actual, source_params,
                                   alpha_h_list, beta_h_list, lr):
    """Data-Knowledge parameter reinforcement for a single sample.

    Paper Eqs. (23)-(30):

        Eq. (23) — H learning frameworks with different (α_h, β_h):
            E_h_T(t) = α_h * (y_T - y_Td)^2  +  β_h * ||Θ_T - K_S||^2
            α_h ∈ [0.8, 1.0],  β_h ∈ [0, 0.2]

        Eq. (24) — Update rule:
            Θ_h_T(t+1) = Θ_T(t) - λ_T * ∂E_h_T / ∂Θ_T

        Eq. (27) — Gradients:
            ∂E/∂w_b = α_h * e_T * v_b  +  β_h * (w_b_T - w_b_S)
            ∂E/∂c_b = α_h * (-2 e_T) * Γ_c  +  β_h * (c_b_T - c_b_S)
            ∂E/∂σ_b = α_h * (-2 e_T) * Γ_σ  +  β_h * (σ_b_T - σ_b_S)
            where Γ_c = w_b * v_b * (x - c_b) / σ_b^2
                  Γ_σ = w_b * v_b * (x - c_b)^2 / σ_b^3

        Eq. (28) — Value function:
            W_h(t) = E[e_h(t), ..., e_h(t+N)]
            (Stability measure over a lookahead window.)

        Eq. (30) — Select best:
            Θ_T(t+1) = Θ_{h*}(t+1)  where  h* = argmin W_h

    We evaluate each (α_h, β_h) candidate, pick the one with lowest
    gradient magnitude (stability proxy), and apply that update.

    Args:
        model         : FNN (target, modified in place)
        x_sample      : ndarray, shape (P,)
        y_actual      : float — ground-truth target
        source_params : dict with 'centers', 'widths', 'weights' from source
        alpha_h_list  : list of α_h values
        beta_h_list   : list of β_h values
        lr            : float — learning rate λ_T
    """
    y_pred, v, u = model.compute_layers(x_sample)
    e_T = y_pred - y_actual
    sum_u = np.sum(u) + EPS

    # Source knowledge parameters (K_S)
    src_c = source_params['centers']   # (K_s, P)
    src_sigma = source_params['widths']  # (K_s, P)
    src_wt = source_params['weights']  # (K_s, 1)

    # Match dimensions: if target has different # rules than source,
    # use the mean source parameters as the knowledge anchor
    K_T = model.n_rules
    if src_c.shape[0] != K_T:
        # Broadcast source mean as knowledge anchor
        anchor_c = np.tile(np.mean(src_c, axis=0), (K_T, 1))
        anchor_sigma = np.tile(np.mean(src_sigma, axis=0), (K_T, 1))
        anchor_w = np.full((K_T, 1), np.mean(src_wt))
    else:
        anchor_c = src_c
        anchor_sigma = src_sigma
        anchor_w = src_wt

    # Evaluate each (α_h, β_h) candidate — Eqs. (23), (27)
    best_stability = float('inf')
    best_grad_w = None
    best_grad_c = None
    best_grad_sigma = None

    for h in range(len(alpha_h_list)):
        alpha_h = alpha_h_list[h]
        beta_h = beta_h_list[h]

        # ── Weight gradient: ∂E/∂w_b = α_h * e_T * v_b + β_h * (w_T - w_S) ──
        grad_w = alpha_h * e_T * v + beta_h * (model.weights - anchor_w)  # (K, 1)

        # ── Center and width gradients ──
        grad_c = np.zeros_like(model.centers)    # (K, P)
        grad_sigma = np.zeros_like(model.widths)  # (K, P)
        safe_widths = np.clip(model.widths, MIN_WIDTH, None)

        for b in range(K_T):
            diff_b = x_sample - model.centers[b]  # (P,)

            # Γ_c = w_b * v_b * (x - c_b) / σ_b^2
            gamma_c = model.weights[b, 0] * v[b, 0] * diff_b / (safe_widths[b] ** 2 + EPS)
            # Γ_σ = w_b * v_b * (x - c_b)^2 / σ_b^3
            gamma_sigma = model.weights[b, 0] * v[b, 0] * (diff_b ** 2) / (safe_widths[b] ** 3 + EPS)

            # Eq. (27): Full gradients with data + knowledge terms
            grad_c[b] = alpha_h * (-2.0 * e_T) * gamma_c + beta_h * (model.centers[b] - anchor_c[b])
            grad_sigma[b] = alpha_h * (-2.0 * e_T) * gamma_sigma + beta_h * (model.widths[b] - anchor_sigma[b])

        # Eq. (28)-(30): Stability measure — use gradient magnitude as proxy
        stability = (np.sum(np.abs(grad_w)) +
                     np.sum(np.abs(grad_c)) +
                     np.sum(np.abs(grad_sigma)))

        if stability < best_stability:
            best_stability = stability
            best_grad_w = grad_w
            best_grad_c = grad_c
            best_grad_sigma = grad_sigma

    # Eq. (24): Apply the best candidate update
    model.weights -= lr * best_grad_w
    model.centers -= lr * best_grad_c
    model.widths -= lr * best_grad_sigma
    model.clip_widths()


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation Metrics (Eqs. 42-44)
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_metrics(y_true, y_pred):
    """Compute evaluation metrics from the paper.

    Eq. (42): RMSE  = sqrt( mean( (y_true - y_pred)^2 ) )
    Eq. (43): sMAPE = mean( 2 * |y_pred - y_true| / (|y_pred| + |y_true|) )
    Eq. (44): MASE  = mean( ((y_pred - y_true) / y_true)^2 )
              Note: The paper defines MASE as a squared relative error metric
              (Eq. 44), which differs from the standard MASE definition used
              in time-series forecasting. We follow the paper's definition.

    Args:
        y_true : ndarray, shape (N,) or (N, 1)
        y_pred : ndarray, shape (N,) or (N, 1)

    Returns:
        rmse  : float
        smape : float
        mase  : float
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()

    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    smape = np.mean(2.0 * np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + EPS))
    mase = np.mean(((yp - yt) / (yt + EPS)) ** 2)

    return rmse, smape, mase


# ═══════════════════════════════════════════════════════════════════════════════
#  Full DK-SOFNN Target Training (Table II)
# ═══════════════════════════════════════════════════════════════════════════════


def train_dk_sofnn(source_model, target_model, X_target_train, y_target_train,
                   source_expert, source_avgs, source_params,
                   N_T=N_T_DEFAULT, lr=TARGET_LR):
    """Full DK-SOFNN target training algorithm (Table II of paper).

    Algorithm flow (inner loop, while t < N_T_samples):
        Every N_T samples:
            Steps 6-9:  Structure adjustment (grow/prune/compensate/constant)
        Every sample:
            Steps 10-13: Parameter reinforcement with (α_h, β_h) selection

    Paper parameters:
        α_h ∈ [0.8, 1.0],  β_h ∈ [0, 0.2]
        λ_T = 0.01,  N_T = 50

    Args:
        source_model    : FNN (trained source, for reference)
        target_model    : FNN (target, initialized from source)
        X_target_train  : ndarray, shape (N, P)
        y_target_train  : ndarray, shape (N, 1)
        source_expert   : dict with 'center', 'width', 'weight'
        source_avgs     : tuple (R̄_S, M̄_S, C̄_S)
        source_params   : dict with 'centers', 'widths', 'weights'
        N_T             : int — structure adjustment interval
        lr              : float — target learning rate λ_T

    Returns:
        rule_history  : list of rule counts at each sample step
        error_history : list of MSE at each structure adjustment
    """
    rule_history = []
    error_history = []
    actions_taken = []
    n_samples = len(X_target_train)

    # Paper Section III-C: H learning frameworks
    alpha_h_list = [0.8, 0.9, 1.0]
    beta_h_list = [0.2, 0.1, 0.0]

    print("=" * 75)
    print("  DK-SOFNN Target Training (Table II)")
    print(f"  Initial rules: {target_model.n_rules} | Samples: {n_samples} | "
          f"N_T: {N_T} | lr: {lr}")
    print("=" * 75)

    for t in range(n_samples):
        # ── Steps 6-9: Structure adjustment every N_T samples ──
        if t > 0 and t % N_T == 0:
            X_seen = X_target_train[:t]
            y_seen = y_target_train[:t]

            action, n_rules, target_avgs = adjust_target_structure(
                target_model, source_avgs, source_expert, X_seen, y_seen
            )
            actions_taken.append(action)

            # Track MSE at this checkpoint
            preds = target_model.predict(X_seen)
            mse = np.mean((preds - y_seen.flatten()) ** 2)
            error_history.append(mse)

            print(f"  [t={t:4d}] {action:30s} | Rules: {target_model.n_rules:3d} | "
                  f"MSE: {mse:.6f}")

        rule_history.append(target_model.n_rules)

        # ── Steps 10-13: Parameter reinforcement (Eqs. 23-30) ──
        x_sample = X_target_train[t]
        y_actual = y_target_train[t, 0]

        parameter_reinforcement_update(
            target_model, x_sample, y_actual,
            source_params, alpha_h_list, beta_h_list, lr
        )

    print(f"\n  Training complete. Final rules: {target_model.n_rules}")
    if actions_taken:
        print(f"  Actions taken: {actions_taken}")
    return rule_history, error_history


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility: Mackey-Glass Synthetic Data
# ═══════════════════════════════════════════════════════════════════════════════


def generate_mackey_glass(n_samples=2000, tau=17, delta_t=1, seed=42):
    """Generate Mackey-Glass chaotic time-series as fallback data.

    The Mackey-Glass equation:
        dx/dt = β * x(t-τ) / (1 + x(t-τ)^10) - γ * x(t)
    with β=0.2, γ=0.1.

    Returns features (4 lags) and targets for regression.
    """
    rng = np.random.RandomState(seed)
    N = n_samples + 1000  # extra for warm-up
    x = np.zeros(N)
    x[0] = 1.2

    beta_mg, gamma_mg = 0.2, 0.1
    for t in range(N - 1):
        t_delayed = max(0, t - tau)
        x[t + 1] = x[t] + delta_t * (
            beta_mg * x[t_delayed] / (1.0 + x[t_delayed] ** 10) - gamma_mg * x[t]
        )

    # Discard warm-up, use last n_samples+10
    x = x[-(n_samples + 10):]
    # Create regression features: 4 lags
    X = np.column_stack([x[i:i + n_samples] for i in range(4)])
    y = x[4:4 + n_samples].reshape(-1, 1)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility: Print Fuzzy Rule Base
# ═══════════════════════════════════════════════════════════════════════════════


def print_rule_base(model, feature_names=None, scaler_X=None, scaler_y=None):
    """Print the final fuzzy rule base in IF-THEN form.

    If scalers are provided, centers and weights are denormalized.
    """
    K = model.n_rules
    P = model.n_inputs
    if feature_names is None:
        feature_names = [f"x{p+1}" for p in range(P)]

    centers = model.centers.copy()
    weights = model.weights.copy()

    if scaler_X is not None:
        centers = scaler_X.inverse_transform(centers)
    if scaler_y is not None:
        weights = scaler_y.inverse_transform(weights)

    print(f"\n{'='*90}")
    print(f"  FINAL FUZZY RULE BASE ({K} Rules)")
    print(f"{'='*90}")
    header = f"{'Rule':>6} | {'IF (Conditions)':<55} | {'THEN Output':>12}"
    print(header)
    print("-" * 90)

    for k in range(K):
        parts = []
        for p in range(P):
            parts.append(f"{feature_names[p]} ≈ {centers[k, p]:.2f}")
        condition = " AND ".join(parts)
        output_val = weights[k, 0]
        print(f"  R{k+1:>3} | {condition:<55} | {output_val:>10.2f}")

    print(f"{'='*90}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN: Full DK-SOFNN Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == '__main__':
    np.random.seed(SEED)

    # ──────────────────────────────────────────────────
    # 1. Load data
    # ──────────────────────────────────────────────────
    feature_names = None
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    dataset_name = None

    try:
        import pandas as pd
        file_path = 'Folds5x2_pp.xlsx'
        df = pd.read_excel(file_path)
        X = df[['AT', 'V', 'AP', 'RH']].values
        y = df['PE'].values.reshape(-1, 1)
        feature_names = ['AT', 'V', 'AP', 'RH']
        dataset_name = "Combined Cycle Power Plant"
        print(f"Loaded {file_path}: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"Could not load Excel data ({e}). Falling back to Mackey-Glass.")
        X, y = generate_mackey_glass(n_samples=2000, seed=SEED)
        feature_names = ['lag1', 'lag2', 'lag3', 'lag4']
        dataset_name = "Mackey-Glass (synthetic)"
        print(f"Generated Mackey-Glass: {X.shape[0]} samples, {X.shape[1]} features")

    # Normalize to [0, 1]
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y)

    # ──────────────────────────────────────────────────
    # 2. Split into source / target (paper setup)
    # ──────────────────────────────────────────────────
    # Paper: 1500 source, 100 target train, 300 target test
    n_source = min(1500, len(X_norm) - 400)
    n_target_train = 100
    n_target_test = min(300, len(X_norm) - n_source - n_target_train)

    X_source = X_norm[:n_source]
    y_source = y_norm[:n_source]

    X_target_train = X_norm[n_source:n_source + n_target_train]
    y_target_train = y_norm[n_source:n_source + n_target_train]

    X_target_test = X_norm[n_source + n_target_train:
                           n_source + n_target_train + n_target_test]
    y_target_test = y_norm[n_source + n_target_train:
                           n_source + n_target_train + n_target_test]

    # Add domain discrepancy: noise N(0, 0.05) on target labels
    noise = np.random.normal(0, 0.05, y_target_train.shape)
    y_target_train_noisy = np.clip(y_target_train + noise, 0, 1)

    print(f"\nDataset: {dataset_name}")
    print(f"  Source samples:       {len(X_source)}")
    print(f"  Target train samples: {len(X_target_train)} (with noise)")
    print(f"  Target test samples:  {len(X_target_test)}")

    n_inputs = X_source.shape[1]

    # ──────────────────────────────────────────────────
    # 3. Train source FNN (Eqs. 7-9)
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 1: Source FNN Training")
    print("=" * 75)
    source_fnn = FNN(n_inputs=n_inputs, n_rules=INITIAL_RULES)
    source_mse = train_source_fnn(source_fnn, X_source, y_source)

    # ──────────────────────────────────────────────────
    # 4. Mine structure knowledge (Eqs. 10-12)
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 2: Structure Knowledge Mining")
    print("=" * 75)
    R_s, M_s, C_s = calculate_indexes(source_fnn, X_source, y_source)
    print(f"  R_S avg (Similarity):   {np.mean(R_s):.4f}")
    print(f"  M_S avg (Sensitivity):  {np.mean(M_s):.4f}")
    print(f"  C_S avg (Contribution): {np.mean(C_s):.4f}")

    # ──────────────────────────────────────────────────
    # 5. Self-organize source FNN (Eqs. 13-15)
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 3: Source Self-Organization")
    print("=" * 75)
    source_fnn = source_self_organization(source_fnn, X_source, y_source,
                                          n_iterations=5)

    # Recompute indexes after self-organization
    R_s, M_s, C_s = calculate_indexes(source_fnn, X_source, y_source)
    source_avgs = (np.mean(R_s), np.mean(M_s), np.mean(C_s))
    print(f"  Post-self-org averages: R={source_avgs[0]:.4f}, "
          f"M={source_avgs[1]:.4f}, C={source_avgs[2]:.4f}")

    # Identify the expert rule (highest contribution C)
    best_rule_idx = np.argmax(C_s)
    source_expert = {
        'center': source_fnn.centers[best_rule_idx].copy(),
        'width': source_fnn.widths[best_rule_idx].copy(),
        'weight': source_fnn.weights[best_rule_idx].copy(),
    }
    source_params = {
        'centers': source_fnn.centers.copy(),
        'widths': source_fnn.widths.copy(),
        'weights': source_fnn.weights.copy(),
    }
    print(f"  Expert rule index: {best_rule_idx}")

    # ──────────────────────────────────────────────────
    # 6. Initialize target FNN from source (knowledge transfer)
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 4: Target FNN Initialization (Knowledge Transfer)")
    print("=" * 75)
    target_fnn = FNN(n_inputs=n_inputs, n_rules=source_fnn.n_rules)
    target_fnn.centers = source_fnn.centers.copy()
    target_fnn.widths = source_fnn.widths.copy()
    target_fnn.weights = source_fnn.weights.copy()
    print(f"  Target initialized with {target_fnn.n_rules} rules from source.")

    # ──────────────────────────────────────────────────
    # 7. Train target with DK-SOFNN (Table II)
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 5: DK-SOFNN Target Training")
    print("=" * 75)
    rule_history, error_history = train_dk_sofnn(
        source_fnn, target_fnn,
        X_target_train, y_target_train_noisy,
        source_expert, source_avgs, source_params,
        N_T=N_T_DEFAULT, lr=TARGET_LR,
    )

    # ──────────────────────────────────────────────────
    # 8. Evaluate and print results (Eqs. 42-44)
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 6: Final Evaluation")
    print("=" * 75)
    y_test_pred = target_fnn.predict(X_target_test).reshape(-1, 1)
    rmse, smape, mase = calculate_metrics(y_target_test, y_test_pred)

    print(f"  Final rule count: {target_fnn.n_rules}")
    print(f"  RMSE  (Eq. 42): {rmse:.6f}")
    print(f"  sMAPE (Eq. 43): {smape:.6f}")
    print(f"  MASE  (Eq. 44): {mase:.6f}")

    # Denormalized RMSE
    y_test_real = scaler_y.inverse_transform(y_target_test)
    y_pred_real = scaler_y.inverse_transform(y_test_pred)
    rmse_real = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))
    print(f"  RMSE (original scale): {rmse_real:.4f}")

    # ──────────────────────────────────────────────────
    # 9. Print the final fuzzy rule base
    # ──────────────────────────────────────────────────
    print_rule_base(target_fnn, feature_names, scaler_X, scaler_y)

    print("\nDK-SOFNN pipeline complete.")
