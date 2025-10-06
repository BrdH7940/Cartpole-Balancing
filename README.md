# Cartpole Balancing using Reinforcement Learning

### 🎯 Problem Definition: CartPole Balancing

**Objective:** Prevent a pole attached to a cart from falling over.

- **State (s):** `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`
- **Action (a):** `0` (push left) or `1` (push right)
- **Reward:** `+1` for every timestep the pole remains upright
- **Termination:** Pole angle > ±12° or cart moves > ±2.4 units from center

---

### 🧮 Mathematical Setup

We'll use **linear function approximation** for both actor and critic.

#### 1. **Feature Engineering**

We'll use polynomial features to capture interactions:

```python
# For state s = [x, x_dot, theta, theta_dot]
# Create quadratic features: [1, x, x_dot, theta, theta_dot, x*theta, x_dot*theta_dot, theta^2, ...]
def get_features(state):
    features = []
    for i in range(len(state)):
        features.append(state[i])
        for j in range(i, len(state)):
            features.append(state[i] * state[j])
    return np.array([1] + features)  # Add bias term
```

Feature vector: `φ(s) ∈ R^d`

#### 2. **Actor: Parameterized Policy**

We use the **softmax policy**:

```python
def policy(features, theta):
    # theta is matrix of size (d × num_actions)
    preferences = features @ theta  # [pref_action0, pref_action1]
    exp_prefs = np.exp(preferences - np.max(preferences))  # Numerical stability
    return exp_prefs / np.sum(exp_prefs)
```

Mathematically:
\[
\pi(a|s; \theta) = \frac{e^{\phi(s)^T \theta*a}}{\sum*{b} e^{\phi(s)^T \theta_b}}
\]
Where `θ ∈ R^(d×2)` are our actor parameters.

#### 3. **Critic: State-Value Function**

Linear function approximation:

```python
def value_function(features, w):
    return features @ w  # scalar
```

\[
V(s; w) = \phi(s)^T w
\]
Where `w ∈ R^d` are our critic parameters.

---

### 🔄 Learning Algorithm: One-Step Actor-Critic

**Update Rules:**

1. **TD Error:**
   \[
   \delta*t = r*{t+1} + \gamma V(s\_{t+1}; w) - V(s_t; w)
   \]

2. **Critic Update (Semi-gradient TD(0)):**
   \[
   w \leftarrow w + \beta \delta_t \nabla_w V(s_t; w) = w + \beta \delta_t \phi(s_t)
   \]

3. **Actor Update (Policy Gradient):**
   \[
   \theta \leftarrow \theta + \alpha \delta*t \nabla*\theta \log \pi(a_t|s_t; \theta)
   \]

Where the **policy gradient** for softmax is:
\[
\nabla*\theta \log \pi(a|s; \theta) = \phi(s)(\mathbf{1}*{a} - \pi(a|s; \theta))
\]
Here `𝟙ₐ` is a one-hot vector for action `a`.
