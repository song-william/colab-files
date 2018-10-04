import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras.models import load_model

def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    model = load_model(args.policy_file)
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    run_bc_model(model, env, max_steps, args.num_rollouts, args.render)

def run_bc_model(model, env, max_steps, num_rollouts, render, verbose=False):

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            action = model.predict(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0 and verbose: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    results = {'returns': returns, 'mean': np.mean(returns), 'std': np.std(returns)}
    return {'observations': np.array(observations), 'actions': np.array(actions)}, results

if __name__ == '__main__':
    main()