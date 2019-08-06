""" eval script """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_episode_reward(filename):
    with open(filename, 'rb') as f:
        episode_results = pickle.load(f)
    rewards = []
    for episode_result in episode_results:
        episode_reward = episode_result['rewards']
        rewards.append(np.mean(np.array(episode_reward)))

    rewards = np.array(rewards)
    rewards_10 = []
    average_length = 1
    for i in range(len(rewards)-average_length):
        rewards_10.append(np.mean(rewards[i:i+average_length]))
    rewards_10 = np.array(rewards_10)
    return rewards_10


if __name__ == '__main__':
    plt.figure(1)

    rewards_1 = load_episode_reward('discrete_2_cases_linear_action_11_argmax_training_noise_0.01.pkl')
    plt.plot(rewards_1, label='discrete_2_cases_linear_action_11_argmax_training_noise_0.01.pkl')

    # rewards_1 = load_episode_reward('./saved_results/coord_conv_x0.2.pkl')
    # rewards_2 = load_episode_reward('./saved_results/random_x0.2.pkl')
    # rewards_3 = load_episode_reward('./saved_results/input_x3.pkl')
    # rewards_4 = load_episode_reward('./saved_results/coord_conv_x0.1.pkl')
    # rewards_5 = load_episode_reward('./saved_results/no_coord_conv.pkl')
    # rewards_6 = load_episode_reward('./saved_results/discrete_action_0_3_wall.pkl')
    # rewards_7 = load_episode_reward('./saved_results/discrete_action_-3_3_wall.pkl')
    # rewards_8 = load_episode_reward('./saved_results/discrete_action_-3_3_wall_footprint.pkl')
    #
    # plt.plot(rewards_1, label='coord_conv_x0.2')
    # plt.plot(rewards_2, label='random_x0.2')
    # plt.plot(rewards_3, label='input_input_input')
    # plt.plot(rewards_4, label='coord_conv_x0.1')
    # plt.plot(rewards_5, label='no_coord_conv')
    # plt.plot(rewards_6, label='discrete_action_0_3_wall')
    # plt.plot(rewards_7, label='discrete_action_-3_3_wall')
    # plt.plot(rewards_8, label='discrete_action_-3_3_wall_footprint')

    plt.xlabel('iter')
    plt.ylabel('episode reward')
    plt.legend()
    plt.show()
