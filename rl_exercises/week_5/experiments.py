import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from policy_gradient import REINFORCEAgent


# Utility to run one episode
def run_single_episode(agent, env, seed=None):
    if seed is not None:
        env.reset(seed=seed)
    state, _ = env.reset()
    done = False
    batch = []
    total_return = 0.0
    while not done:
        action, info = agent.predict_action(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        batch.append((state, action, float(reward), next_state, done, info))
        state = next_state
        total_return += reward
    if hasattr(agent, "update_agent"):
        agent.update_agent(batch)
    return total_return


def run_reinforce_experiments(env_name, builders, num_episodes, seeds):
    results = {}
    for name, build in builders.items():
        print(f"Config: {name}")
        all_returns = []
        for seed in seeds:
            env = gym.make(env_name)
            agent = build(env, seed)
            returns = []
            for ep in range(1, num_episodes + 1):
                ret = run_single_episode(agent, env, seed)
                returns.append(ret)
                if ep % max(1, num_episodes // 10) == 0:
                    print(f"  Episode {ep}/{num_episodes}: return={ret:.2f}")
            all_returns.append(returns)
            print(f"Finished seed {seed}), avg return = {np.mean(returns):.2f}\n")
        results[name] = np.array(all_returns)  # shape (n_seeds, episodes)
    return results


# Plotting
def plot_curve(name, data, output_dir):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    n = data.shape[0]
    ci95 = 1.96 * std / np.sqrt(n)
    episodes = len(mean)
    plt.figure()
    plt.plot(np.arange(1, episodes + 1), mean, label="Mean")
    plt.fill_between(np.arange(1, episodes + 1), mean - ci95, mean + ci95, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{name} Learning Curve")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"{name}_learning_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved plot for {name} -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run REINFORCE experiments and compare to DQN"
    )
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Gym environment ID"
    )
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of episodes per seed"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1], help="Up to 2 random seeds"
    )
    args = parser.parse_args()

    output_dir = "plot"
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Output directory: {output_dir}\nUsing seeds: {args.seeds}\nEpisodes per run: {args.episodes}\n"
    )

    # REINFORCE configurations
    builders = {}
    for ms in [100, 200]:
        builders[f"Traj{ms}"] = (
            lambda ms: lambda env, sd: REINFORCEAgent(
                env=gym.wrappers.TimeLimit(env, max_episode_steps=ms),
                lr=1e-2,
                gamma=0.99,
                seed=sd,
            )
        )(ms)
    combos = [(32, 1e-2), (128, 1e-2), (256, 1e-3), (256, 1e-2)]
    for h, lr in combos:
        builders[f"h{h}_lr{lr}"] = (
            lambda h, lr: lambda env, sd: REINFORCEAgent(
                env=env, lr=lr, gamma=0.99, seed=sd, hidden_size=h
            )
        )(h, lr)

    reinforce_results = run_reinforce_experiments(
        args.env, builders, args.episodes, args.seeds
    )
    for name, data in reinforce_results.items():
        plot_curve(name, data, output_dir)

    print("All experiments done.")
