#!/usr/bin/env python3
'''
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3 run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of the original vesion of this script and included expert policies: Jonathan Ho(hoj@openai.com)

Author of modified script: Alok Singh (alokbeniwal@gmail.com)
'''

import argparse

import gym
import h5py
import keras
import numpy as np
import tensorflow as tf

import tf_util
from load_policy import load_policy

parser = argparse.ArgumentParser()

parser.add_argument(
    'expert_policy_file', type=str, default='experts/Ant-v1.pkl')
parser.add_argument('envname', type=str, default='Ant-v1')
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument("--test", action='store_false')  # test by default
parser.add_argument("--train", action='store_true')  # train if flag passed
parser.add_argument(
    '--num_rollouts', type=int, default=20, help='Number of expert roll outs')
parser.add_argument('-m', '--model', type=str, default='my_model.h5')
parser.add_argument('-o', '--output', type=str, default='my_model.h5')
parser.add_argument('--dagger', action='store_true')

args = parser.parse_args()


def main():
    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        expert_policy = load_policy(args.expert_policy_file)
        imitation_policy = keras.models.load_model(args.model)

        for i in range(args.num_rollouts):
            print('iter', i)

            obs = env.reset()
            done = False
            totalr = 0
            steps = 0

            while not done:

                action = imitation_policy.predict(obs[None, :])

                observations.append(obs)

                # If running DAgger, use expert's actions and aggregate. Else use imitation learner.
                if args.dagger:
                    expert_action = expert_policy(obs[None, :])
                    actions.append(expert_action)
                else:
                    actions.append(action)

                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {
            'observations': np.array(observations),
            'actions': np.array(actions)}

        if args.train or args.dagger:
            with h5py.File('obs.h5', 'w') as hf:
                hf.create_dataset('obs', data=np.array(observations))
            with h5py.File('act.h5', 'w') as hf:
                hf.create_dataset('act', data=np.array(actions))

        return expert_data


if __name__ == '__main__':
    main()
