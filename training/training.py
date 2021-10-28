# agent[0]:adversary, initial position is (0,0), it's positon varible: env.world.agents[0].state.p_pos
# agent[1]:agent, initial position is (0.5,0.5), it's position varible：env.world.agents[1].state.p_pos
# checkpoint's position is (-0.5,-0.5), it's position varible：env.world.check[index].state.p_pos
# landmarks' position: (0.35, 0.35), (0.35,-0.35), (0, -0.35)
# checkpoint's position: (-0.5, -0.5)

#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import time

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from RL_brain_training import QLearningTable

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def action_switch(action):
    if 0 == action:
        return np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], dtype=float)
    elif 1 == action:
        return np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=float)
    elif 2 == action:
        return np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=float)
    elif 3 == action:
        return np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=float)
    elif 4 == action:
        return np.array([[1, 0, 0, 0, 0], [0, 0, 0.70710678, 0.70710678, 0]], dtype=float)
    elif 5 == action:
        return np.array([[1, 0, 0, 0, 0], [0, 0, 0.70710678, 0, 0.70710678]], dtype=float)
    elif 6 == action:
        return np.array([[1, 0, 0, 0, 0], [0, 0.70710678, 0, 0.70710678, 0]], dtype=float)
    elif 7 == action:
        return np.array([[1, 0, 0, 0, 0], [0, 0.70710678, 0, 0, 0.70710678]], dtype=float)

def direction_switch(action):
    if 0 == action:
        return np.array([0.0, 1.0], dtype=float)
    elif 1 == action:
        return np.array([0.0, -1.0], dtype=float)
    elif 2 == action:
        return np.array([-1.0, 0.0], dtype=float)
    elif 3 == action:
        return np.array([1.0, 0.0], dtype=float)
    elif 4 == action:
        return np.array([-0.70710678, 0.70710678], dtype=float)
    elif 5 == action:
        return np.array([-0.70710678, -0.70710678], dtype=float)
    elif 6 == action:
        return np.array([0.70710678, 0.70710678], dtype=float)
    elif 7 == action:
        return np.array([0.70710678, -0.70710678], dtype=float)

def Normalization(array):
    sum = array[0]**2 + array[1]**2
    temp = np.sqrt(1/sum)
    return np.array([array[0]*temp, array[1]*temp])

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_tag.py', help='Path of the scenario Python script.')
    args = parser.parse_args()
    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)
    # execution loop
    obs_n = env.reset()
    # render call to create viewer window (necessary only for interactive policies)
    target = env.world.check[0].state.p_pos

    # set the action space
    action_space = ['up', 'down', 'left', 'right', 'lu', 'ld', 'ru', 'rd']
    n_actions = len(action_space)
    # set the parameter
    RL = QLearningTable(actions=list(range(n_actions)), learning_rate=0.4, reward_decay=0.999)
    ratio = 30
    constringency = 0

    for episode in range(10000):
        # save the q-table data per one hundred episodes
        if 0 == (episode % 100):
            RL.q_table.to_json("../display/q_table_30.json")

        obs_n = env.reset()
        done = False
        start_time = time.clock()
        collision_number = 0

        # the gradients of epsilon_
        if episode < 800:
            epsilon_ = 0.5
        elif episode < 1400:
            epsilon_ = 0.6
        elif episode < 1800:
            epsilon_ = 0.7
        elif episode < 2100:
            epsilon_ = 0.8
        elif episode < 2300:
            epsilon_ = 0.9
        elif episode < 2400:
            epsilon_ = 0.95
        elif episode < 6000:
            epsilon_ = 0.99
        else:
            epsilon_ = 1

        while True:
            act_n = []
            act_d = []
            reward = 0.0

            dist = np.sqrt(np.sum(np.square(env.world.agents[1].state.p_pos - env.world.check[0].state.p_pos)))

            # agent choose the action based on the position message
            agent1_obs = [int(env.world.agents[1].state.p_pos[0] * ratio), int(env.world.agents[1].state.p_pos[1] * ratio)]
            action = RL.choose_action(str(agent1_obs), epsilon_)
            act_n = action_switch(action)
            act_d = direction_switch(action)

            # step environment
            obs_n_, reward_n_, done_n_, info_n_ = env.step(act_n, target)

            # p_pos_: the next position that the agent will be
            # dist 为上一时刻距离checkpoint的距离, dist_ 为下一时刻距离checkpoint的距离
            p_pos_ = np.array([obs_n_[1][0],obs_n_[1][1]])
            agent2check = Normalization(world.check[0].state.p_pos - p_pos_)
            dist_ = np.sqrt(np.sum(np.square(p_pos_ - world.check[0].state.p_pos)))


            if (scenario.is_collision(env.world.agents[0], env.world.agents[1])):
                collision_number += 1
            if (scenario.is_collision(env.world.agents[1], env.world.landmarks[0]) or
                    scenario.is_collision(env.world.agents[1], env.world.landmarks[1]) or
                    scenario.is_collision(env.world.agents[1], env.world.landmarks[2])):
                reward = -1000
                done_str = "Failure!"
                done = True
            elif (env.world.agents[1].state.p_pos[0] > 0.6 or
                  env.world.agents[1].state.p_pos[0] < -0.6 or
                  env.world.agents[1].state.p_pos[1] > 0.6 or
                  env.world.agents[1].state.p_pos[1] < -0.6):
                reward = -1000
                done_str = "Failure!"
                done = True
            elif (scenario.is_done(env.world.agents[1], env.world)):
                reward = 10000
                done_str = "Success!"
                done = True
            else:
                done = False

            reward += dist_ * (np.dot(agent2check, act_d) - 1.0)

            # agent1_obs 为当前位置观测值，而agent1_obs_ 为下一状态位置观测值
            agent1_obs_ = [int(obs_n_[1][0]*ratio),int(obs_n_[1][1]*ratio)]
            RL.learn(str(agent1_obs), action, reward, str(agent1_obs_),done)
            # swap the observation
            obs_n = obs_n_

            if done:
                end_time = time.clock()
                operating_time = end_time - start_time
                print("第",episode,"次测试: " + done_str, "，碰撞次数为：", collision_number, "使用时间为：", operating_time, " seconds.")
                if done_str == "Success!":
                    constringency += 1
                elif done_str == "Failure!":
                    constringency = 0
                break

        if constringency == 10:
            print("the q-table has been convergenced!")
            print("You can python display.py to see the effect of q-learning")
            break
