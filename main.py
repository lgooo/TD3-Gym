import numpy as np
import torch
import gym
import argparse
import os
from config import Config
import utils
import TD3


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Walker2d-v2")
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    config = Config()

    file_name = f"TD3_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if config.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": config.discount,
        "tau": config.tau,
    }

    # Initialize policy
    kwargs["policy_noise"] = config.policy_noise * max_action
    kwargs["noise_clip"] = config.noise_clip * max_action
    kwargs["policy_freq"] = config.policy_freq
    policy = TD3.TD3(**kwargs)

    if config.load_model != "":
        policy_file = file_name if config.load_model == "default" else config.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    rewards = []
    visited_states = []

    random_peroid = 10000
    random_steps = 1000

    for t in range(int(config.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * config.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        visited_states.append(state)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= config.start_timesteps:
            policy.train(replay_buffer, config.batch_size)

        if done:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            rewards.append(episode_reward)
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


    env.close()
    np.save(f"./results/{file_name}_rewards", rewards)
    np.save(f"./results/{file_name}_states", visited_states)
