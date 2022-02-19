"""
Example script to visualize or evaluate your trained agent.
This file might be useful in debugging.
"""
import gym
import numpy as np
from load_agents import PolicyAPI
from collections import deque
from core.utils import summary, save_progress
from pprint import pprint

def evaluate_model(env_name, log_dir, suffix, num_episodes):
    # env_name = "MetaDrive-Tut-10Env-v0"
    policy = PolicyAPI(
        env_name,  # In order to get the observation shape
        num_envs=1,
        log_dir=log_dir,
        suffix=suffix
    )
    print("======= Start Evaluating in Env {}! ==========".format(env_name))

    # Turn use_render to False if you are trying to evaluate agents.
    comp_env = gym.make(env_name, config={'use_render': False})
    obs = comp_env.reset()

    ep_rew = 0.0
    reward_recorder = deque(maxlen=num_episodes)
    success_recorder = deque(maxlen=num_episodes)
    episode_count = 0

    while True:
        # frame = comp_env.render()
        act = policy(obs)
        # print("Current step: {}, Action: {}".format(i, act))
        obs, rew, term, info = comp_env.step(act)
        ep_rew += rew
        if term:
            success = info["arrive_dest"]
            success_recorder.append(float(success))
            reward_recorder.append(float(ep_rew))
            print("Episode {}, reward: {}, success rate: {}".format(episode_count + 1, ep_rew,
                                                                    np.mean(success_recorder)))
            ep_rew = 0
            episode_count += 1
            obs = comp_env.reset()
            if episode_count >= num_episodes:
                break
    comp_env.close()

    stats = dict(
        training_episode_reward=summary(reward_recorder, "episode_reward"),
        success_rate=summary(success_recorder, "success_rate"),
        env_name=env_name,
        log_dir=log_dir
    )
    return stats

if __name__ == '__main__':

    train_env = "MetaDrive-Tut-Test-v0"
    log_dir = "./MetaDrive50Env/PPO"
    suffix = "best"
    num_episodes = 51
    train_stats = evaluate_model(train_env, log_dir, suffix, num_episodes)
    pprint(train_stats)
