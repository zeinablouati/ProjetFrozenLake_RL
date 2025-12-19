import numpy as np
from typing import Tuple, List
from gymnasium import Env


def sarsa(
    env: Env,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    SARSA tabulaire (on-policy)

    Returns
    -------
    Q : np.ndarray
        Table Q (n_states, n_actions)
    rewards : list[float]
        Récompense totale par épisode
    """

    if seed is not None:
        np.random.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    episode_rewards: List[float] = []

    for episode in range(n_episodes):
        state, _ = env.reset()

        # Action initiale (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Prochaine action (on-policy)
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            # Mise à jour SARSA
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            action = next_action
            total_reward += reward

        episode_rewards.append(total_reward)

    return Q, episode_rewards
