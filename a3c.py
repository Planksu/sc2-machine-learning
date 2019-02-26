# Implementation for the asynchronous actor-critic algorithm

import os
import sys, getopt
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from time import time, sleep, gmtime, strftime
import Queue
import random

random.seed(100)

# Flags
T_MAX = 1000
NUM_THREADS = 16
INITIAL_LEARNING_RATE = 1e-5
DISCOUNT_FACTOR = 0.99
VERBOSE_EVERY = 40000
TESTING = False

I_ASYNC_UPDATE = 5

FLAGS = {	"T_MAX": T_MAX, "NUM_THREADS": NUM_THREADS, "INITIAL_LEARNING_RATE": INITIAL_LEARNING_RATE,
			"DISCOUNT_FACTOR": DISCOUNT_FACTOR, "VERBOSE_EVERY": VERBOSE_EVERY, "TESTING": TESTING, "I_ASYNC_UPDATE": I_ASYNC_UPDATE}

training_finished = False

class Summary:
	def __init__(self, logdir, agent):
		with tf.variable_scope('summary'):
			summarising 		= ['episode_avg_reward', 'avg_value']
			self.agent 			= agent
			self.writer 		= tf.summary.FileWriter(logdir, self.agent.sess.graph)
			self.summary_ops 	= {}
			self.summary_vars 	= {}
			self.summary_ph 	= {}
			for s in summarising:
				self.summary_vars[s] 	= tf.Variable(0.0)
				self.summary_ph[s] 		= tf.placeholder('float32', name=s)
				self.summary_ops[s]		= tf.summary.scalar(s, self.summary_vars[s])
			self.update_ops = []
			for k in self.summary_vars:
				self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
			self.summary_op = tf.summary.merge(list(self.summary_ops.values()))

	def write_summary(self, summary, t):
		self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in summary.items()})
		summary_to_add = self.agent.sess.run(self.summary_op, {self.summary_vars[k]: v for k, v in summary_items()})
		self.writer.add_summary(summary_to_add, global_step=t)

def async_trainer(agent, env, sess, thread_idx, T_queue, summary, saver, save_path):
	print("Training thread", thread_idx)
	T = T_queue.get()
	T_queue.put(T+1)
	t = 0

	last_verbose = T
	last_time = time()
	last_target_update = T

	while T < T_MAX:
		t_start = t
		batch_states = []
		batch_rewards = []
		batch_actions = []
		baseline_values = []

		while len(batch_states) < I_ASYNC_UPDATE:
			batch_states.append(state)

			policy, value = agent.get_policy_and_value(state)
			action_idx = np.random.choice(agent.action_size, p=policy)

			#state, reward

			t += 1
			T = T_queue.get()
			T_queue.put(T+1)

			reward = np.clip(reward, -1, 1)

			batch_rewards.append(reward)
			batch_actions.append(action_idx)
			baseline_values.append(value[0])

		target_value = agent.get_value(state)[0]
		last_R = target_value

		batch_target_values = []
		for reward in reversed(batch_rewards):
			target_value = reward + DISCOUNT_FACTOR * target_value
			batch_target_values.append(target_value)

		batch_target_values.reverse()

		if TESTING:
			temp_rewards = batch_rewards + [last_R]
			test_batch_target_values = []
			for j in range(len(batch_rewards)):
				test_batch_target_values.append(discount(temp_rewards[j:], DISCOUNT_FACTOR)) 
			if not test_equals(batch_target_values, test_batch_target_values, 1e-5):
				print("Assertion failed")

		batch_advantages = np.array(batch_target_values) - np.array(baseline_values)

		agent.train(np.vstack(batch_states), batch_actions, batch_target_values, batch_advantages)

	global training_finished
	training_finished = True

def estimate_reward(agent, episodes=10, max_steps=10000):
	episode_rewards = []
	episode_vals = []
	t = 0
	for i in range(episodes):
		episode_reward = 0
		policy, value = agent.get_policy_and_value(state)
		action_idx = np.random.choice(agent.action_size, p=policy)
		t += 1
		episode_vals.append(value)
		episode_reward += reward
		if t > max_steps:
			episode_rewards.append(episode_reward)
			return episode_rewards, episode_vals

		episode_rewards.append(episode_reward)
	return episode_rewards, episode_vals


def evaluator(agent, T_queue, summary, saver, save_path):
    T = T_queue.get()
    T_queue.put(T)
    last_time = time()
    last_verbose = T
    while T < T_MAX:
        T = T_queue.get()
        T_queue.put(T)
        if T - last_verbose >= VERBOSE_EVERY:
            print "T", T
            current_time = time()
            print "Train steps per second", float(T - last_verbose) / (current_time - last_time)
            last_time = current_time
            last_verbose = T

            print "Evaluating agent"
            episode_rewards, episode_vals = estimate_reward(agent, env, episodes=5)
            avg_ep_r = np.mean(episode_rewards)
            avg_val = np.mean(episode_vals)
            print "Avg ep reward", avg_ep_r, "Average value", avg_val

            summary.write_summary({'episode_avg_reward': avg_ep_r, 'avg_value': avg_val}, T)
            checkpoint_file = saver.save(sess, save_path, global_step=T)
            print "Saved in", checkpoint_file
        sleep(1.0)


def a3c(game_name, num_threads=16, restore=None, save_path='model'):
    processes = []
    envs = []
    for _ in range(num_threads+1):
        #start games

    # Separate out the evaluation environment
    evaluation_env = envs[0]
    envs = envs[1:]

    with tf.Session() as sess:
        agent = Agent(session=sess,
        action_size=envs[0].action_size, model='mnih',
        optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))

        # Create a saver, and only keep 2 checkpoints.
        saver = tf.train.Saver(max_to_keep=2)

        T_queue = Queue.Queue()
        summary = Summary(save_path, agent)

        # Create a process for each worker
        for i in range(num_threads):
            processes.append(threading.Thread(target=async_trainer, args=(agent,
            envs[i], sess, i, T_queue, summary, saver, save_path,)))

        # Create a process to evaluate the agent
        processes.append(threading.Thread(target=evaluator, args=(agent,
        evaluation_env, sess, T_queue, summary, saver, save_path,)))

        # Start all the processes
        for p in processes:
            p.daemon = True
            p.start()

        # Until training is finished
        while not training_finished:
            sleep(0.01)

        # Join the processes, so we get this thread back.
        for p in processes:
            p.join()

# Returns sum(rewards[i] * gamma**i)
def discount(rewards, gamma):
    return np.sum([rewards[i] * gamma**i for i in range(len(rewards))])

def test_equals(arr1, arr2, eps):
    return np.sum(np.abs(np.array(arr1)-np.array(arr2))) < eps

def main(argv):
    num_threads = NUM_THREADS
    save_path = None
    restore = None
    try:
        opts, args = getopt.getopt(argv, "hg:s:r:t:")
    except getopt.GetoptError:
        print "To run the game and save to the given save path: \
        a3c.py -s <save path> -t <num threads>"
    for opt, arg in opts:
        if opt == '-h':
            print "Options: -s <save path>, -t <num threads>."
            sys.exit()
        elif opt == '-s':
            save_path = arg
        elif opt == '-t':
            num_threads = int(arg)
            print "Using", num_threads, "threads."
    if save_path is None:
        save_path = 'experiments/' + game_name + '/' + \
        strftime("%d-%m-%Y-%H:%M:%S/model", gmtime())
        print "No save path specified, so saving to", save_path
    if not os.path.exists(save_path):
        print "Path doesn't exist, so creating"
        os.makedirs(save_path)
    print "Using save path", save_path
    print "Using flags", FLAGS
    a3c(game_name, num_threads=NUM_THREADS, restore=restore,
    save_path=save_path)

if __name__ == "__main__":
main(sys.argv[1:])
