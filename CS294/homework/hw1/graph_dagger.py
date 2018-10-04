import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras.models import load_model
from run_bc_policies import run_bc_model
import json
import matplotlib.pyplot as plt

with open('dagger_graph_data.json') as f:	
    data = json.load(f)

num_iterations = (data['total_rollouts'] - data['expert_rollouts'])//data['iteration_rollout'] + 1
iterations = list(range(num_iterations))

# plot dagger results
means = [x['mean'] for x in data['dagger_results']]
stds = [x['std'] for x in data['dagger_results']]
plt.errorbar(iterations, means, yerr=stds, fmt='o', linestyle='-.', label="dagger")

# plot expert results
means, stds = [data['expert_results']['mean']]*len(iterations), [data['expert_results']['std']]*len(iterations) 
plt.errorbar(iterations, means, yerr=stds, linestyle=':', label="expert")

# plot behavioral clone results
means, stds = [data['bc_results']['mean']]*len(iterations), [data['bc_results']['std']]*len(iterations)
print('stds:', stds[0])
plt.errorbar(iterations, means, yerr=stds, linestyle=':', label="behavioral cloning") 
plt.xlabel("Dagger Iterations")
plt.ylabel("Avg Reward")
plt.title("Performance of Dagger on {}".format('Hopper-v2'))
plt.legend(loc='lower right')
plt.show()
