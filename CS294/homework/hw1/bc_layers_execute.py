import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras.models import load_model
import matplotlib.pyplot as plt

def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    all_returns = []
    means = []
    stds = []
    for i in range(10):
        print("running hidden layers", i)
        model = load_model("experimental/{}_{}.h5".format(args.envname, i))
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            # print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model.predict(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        mean = np.mean(returns)
        std = np.std(returns)
        
        returns.append(returns)
        means.append(mean)
        stds.append(std)
        print('returns', returns)
        print('mean return', mean)
        print('std of return', std)
    
    print("means:", means)
    print("stds:", stds)
    iterations = list(range(1, 11))
    plt.errorbar(iterations, means, yerr=stds, fmt='o', linestyle='-.')
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Avg Reward")
    plt.title("Performance of Behavioral Cloning on {} vs Number of NN layers".format(args.envname))
    plt.show()
if __name__ == '__main__':
    main()