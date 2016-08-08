# Reinforce learning example
# - Code example from http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/index.html lab1.
# - Referred to this cem.py project: https://gist.github.com/sjb373/6502fbf55ecdf988aa247ef7f60a9546 
# 
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt

# ================================================================
# Policies
# ================================================================

class DeterministicDiscreteActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        #assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0 : dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions : None].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        print("ob ", ob)
        print("W ",self.W)
        print("b ",self.b)
        y = ob.dot(self.W) + self.b
        print("y ", y)
        a = y.argmax()
        print("a ", a)
        return a

class DeterministicContinuousActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        #assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0 : dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac : None]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a

def do_episode(policy, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = policy.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew

env = None
def noisy_evaluation(theta):
    policy = make_policy(theta)
    rew = do_episode(policy, env, num_steps)
    return rew

def make_policy(theta):
    if isinstance(env.action_space, Discrete):
        return DeterministicDiscreteActionLinearPolicy(theta,
            env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return DeterministicContinuousActionLinearPolicy(theta,
            env.observation_space, env.action_space)
    else:
        raise NotImplementedError

# Task settings:
env = gym.make('CartPole-v0') # Change as needed
num_steps = 500 # maximum length of episode
# Alg settings:
n_iter = 3 # number of iterations of CEM
batch_size = 25 # number of samples per batch
elite_frac = 0.2 # fraction of samples used as elite set

if isinstance(env.action_space, Discrete):
    dim_theta = (env.observation_space.shape[0]+1) * env.action_space.n
elif isinstance(env.action_space, Box):
    dim_theta = (env.observation_space.shape[0]+1) * env.action_space.shape[0]
else:
    raise NotImplementedError

# Initialize mean and standard deviation
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)

# (One of?) Good CartPole result
#theta_mean = np.array([0.2226394,  0.37167546,-0.23130909,-0.51706018,-3.84031352, 0.40085067,-2.87828687, 0.47947236 ,0.20460138 ,0.19369949])
#theta_std = np.array([  7.37634136e-03,  9.64920053e-03,   6.17346100e-04,   8.99188496e-03,
#            3.63448041e-02,   3.40082449e-02,   5.78441799e-04,   2.09394903e-03,
#                3.15217928e-04,   3.67459568e-03])

theta_mean_result = np.empty([n_iter,dim_theta])
theta_std_result = np.empty([n_iter,dim_theta])

# Now, for the algorithm
for iteration in range(n_iter):
    # Sample parameter vectors
    thetas = np.zeros((batch_size, dim_theta))
    for i in range(dim_theta):
        thetas[:,i]=np.random.normal(loc=theta_mean[i], scale=theta_std[i], size=(batch_size,))
    rewards = [noisy_evaluation(theta) for theta in thetas]
    # Get elite parameters
    n_elite = int(batch_size * elite_frac)
    elite_inds = np.argsort(rewards)[batch_size - n_elite:batch_size]
    elite_thetas = [thetas[i] for i in elite_inds]
    elite_array = np.asarray(elite_thetas)
    #print(elite_array)
    # Update theta_mean, theta_std
    for i in range(dim_theta):
        theta_mean[i] = elite_array[:,i].mean()
        theta_std[i] = elite_array[:,i].std()
    print("iteration %i. mean f: %8.3g. max f: %8.3g "%(iteration, np.mean(rewards), np.max(rewards)))
    theta_mean_result[iteration,:] = theta_mean
    theta_std_result[iteration,:] = theta_std
    do_episode(make_policy(theta_mean), env, num_steps, render=True)

plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(0,n_iter,1), theta_mean_result)
plt.subplot(212)
plt.plot(np.arange(0,n_iter,1), theta_std_result)
plt.show()
print(theta_mean_result)
print(theta_std_result)
