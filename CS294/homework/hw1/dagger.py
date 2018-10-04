import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from train_bc_models import train_bc, load_data
from run_bc_policies import run_bc_model
import json
from run_expert import run_expert

def main():
    # filename = 'Ant-v2'
    expert_rollouts = 12
    iteration_rollout = 1
    total_rollouts = 20
    experiment_rollouts = 20
    hidden_layers = 2
    epochs = 300

    results = []

    envname = 'Hopper-v2'
    expert_dagger_data = 'expert_dagger_data'
    
    # os.system('python3 run_expert.py experts/{}.pkl {} --num_rollouts={} --save_dir={}'.format(envname, envname, expert_rollouts, expert_dagger_data))
    run_expert('experts/{}.pkl'.format(envname), envname, num_rollouts=expert_rollouts, save_dir=expert_dagger_data)

    filepath = '{}/{}.pkl'.format(expert_dagger_data, envname)
    with open(filepath, 'rb') as f:
        data = pickle.loads(f.read())

    x_train = np.array(data['observations'])
    y_train = np.array(data['actions'])
    # reshape y
    x, _, y = y_train.shape
    y_train = np.reshape(y_train, (x, y))

    with tf.Session():
        tf_util.initialize()
                
        policy_fn = load_policy.load_policy("experts/{}.pkl".format(envname))

        model = train_bc(x_train, y_train, envname, hidden_layers=hidden_layers, epochs=epochs)

        for i in range(expert_rollouts, total_rollouts, iteration_rollout):
            
            print("iteration:", i)

            env = gym.make(envname)
            max_steps = env.spec.timestep_limit
            data, _ = run_bc_model(model, env, max_steps, iteration_rollout, False)

            _, result = run_bc_model(model, env, max_steps, experiment_rollouts, False)
            results.append(result)

            actions = np.array([policy_fn(obs[None,:]) for obs in data['observations']])

            # reshape y
            x, _, y = actions.shape
            y_train_temp = np.reshape(actions, (x, y))

            x_train = np.concatenate((x_train, np.array(data['observations'])))
            y_train = np.concatenate((y_train, y_train_temp))
            model = train_bc(x_train, y_train, envname, hidden_layers=hidden_layers, epochs=epochs)

        model.save("dagger_policies/{}-{}-{}.h5".format(envname, expert_rollouts, total_rollouts))
        env = gym.make(envname)
        max_steps = env.spec.timestep_limit
        data, result = run_bc_model(model, env, max_steps, experiment_rollouts, False)
        results.append(result)
   
    with tf.Session():
        tf_util.initialize()
        expert_results = run_expert('experts/{}.pkl'.format(envname), envname, num_rollouts=total_rollouts)

    with tf.Session():
        tf_util.initialize()
        env = gym.make(envname)
        max_steps = env.spec.timestep_limit
        x_train, y_train = load_data('expert_data/{}.pkl'.format(envname))
        model = train_bc(x_train, y_train, envname, hidden_layers=hidden_layers, epochs=epochs)
        _, bc_results = run_bc_model(model, env, max_steps, experiment_rollouts, False)

    final_data = {
                'expert_rollouts': expert_rollouts,
                'iteration_rollout': iteration_rollout,
                'total_rollouts': total_rollouts,
                'dagger_results': results,
                'expert_results': expert_results,
                'bc_results': bc_results
                }
    
    with open('dagger_graph_data.json', 'w') as outfile:
        json.dump(final_data, outfile)

if __name__ == '__main__':
    main()
