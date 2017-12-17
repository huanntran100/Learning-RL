# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import seaborn as sns
import gym


def evaluate_policy(env, gamma, policy, max_iterations, tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    # Initialize value function 
    value_function = np.zeros(env.nS)
    for _ in range(max_iterations): 
        delta = 0
        for s in range(env.nS): 
            v = 0
            # Policy (and action_prob) decides how the agent acts probability wise 
            for a, action_prob in enumerate(policy[s]): 
                # Prob accounts for stochastic dynamics of the envs - 1.0 in deterministic envs 
                # If no stochasticity, then i only goes up to 0 and you update v only once for every action. 
                for i, (prob, next_state, reward, done) in enumerate(env.P[s][a]):
                    v += action_prob * prob * (reward + gamma * value_function[next_state])  
            delta = np.maximum(delta, np.abs(v - value_function[s])) 
            value_function[s] = v 
        if delta < tol:
            break     
    return value_function, _ 


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """   
    policy_function = []
    for s in range(env.nS): 
        action_values = np.zeros(env.nA)
        for a in range(env.nA): 
            for i, (prob, next_state, reward, done) in enumerate(env.P[s][a]):
                action_values[a] += prob * (reward + gamma * value_function[next_state]) 
        best_action = np.argmax(action_values) 
        policy_function.append(best_action)
    return policy_function 


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
    
    policy_function = value_function_to_policy(env, gamma, value_func)
    
    for s in range(env.nS):
        old_action = np.argmax(policy[s])  
        best_action = policy_function[s]

        
        if old_action != best_action:
            policy_stable = False 
        
        policy[s] = np.eye(env.nA)[best_action]  
    return policy_stable, policy


def policy_iteration(env, gamma, max_iterations, tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    # Initialize value function with all zeros    
    value_func = np.zeros(env.nS) 
    
    # Initialize random policy  
    policy = np.ones([env.nS, env.nA]) / env.nA
   
    # Initialize counts for policy evaluation and improvement 
    eval_count = 0
    improve_count = 0 
      
    while True: 
        # Policy Evaluation  
        value_func, num_evals = evaluate_policy(env, gamma, policy, max_iterations, tol) 
        
        # Policy Improvement  
        is_stable, policy = improve_policy(env, gamma, value_func, policy)
        
        # Update counts 
        eval_count += num_evals 
        improve_count += 1 
        
        # If the policy is stable, exit
        if is_stable == True:
            break
    
    return policy, value_func, eval_count, improve_count 


def value_iteration(env, gamma, max_iterations, tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    
    value_func = np.zeros(env.nS) 
    while True: 
    #for _ in range(max_iterations):  
        delta = 0
        _ = 0
        for s in range(env.nS):
            v = value_func[s]
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    # Sum over transitions and take the maximum 
                    action_values[a] += prob * (reward + gamma * value_func[next_state]) 
            # Take maximum transition value
            best_Action = max(action_values)
            delta = max(delta, np.abs(best_Action - v))
            value_func[s] = best_Action 
            _ += 1
        if delta < tol:
            break 
    return value_func, _ 
       

def execute_policy(env, gamma, policy_func):
    """Execute policy using agent and prints discounted cumulative reward. 

    Parameters
    ----------
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.ndarray
      Array of state to action number mappings
    """
    total_reward = 0 
    _ = 0
    observation = env.reset()
    while True: 
        action = np.argmax(policy_func[observation])
        observation, reward, done, info = env.step(action)
        total_reward += np.power(gamma,_) * reward 
        _ += 1
        if done:
            break
    print('Total discounted cumulative reward: ', total_reward) 
       
        
        

    
def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = []
    #Assuming a square map
    map_dim = int(np.sqrt(len(policy)))
    for s in range(len(policy)): 
        dictKey = np.argmax(policy[s])
        str_policy.append(action_names[dictKey][:1])
    str_policy = np.reshape(str_policy, (map_dim, map_dim))
    print(str_policy)
    
def print_value_function(value_func): 
    """Plot the value function using a heat map.

    Parameters
    ----------
    value_func: np.ndarray
      Array of state values
    """
    map_dim = int(np.sqrt(len(value_func)))
    value_plot = np.reshape(value_func, (map_dim, map_dim)) 
    sns.heatmap(value_plot)


    
