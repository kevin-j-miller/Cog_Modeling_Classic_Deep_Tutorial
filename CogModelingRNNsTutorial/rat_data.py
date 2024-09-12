"""Functions for loading rat data."""
import json
import os
from typing import List
import urllib.request

import numpy as np
import scipy.io

from Cog_Modeling_Classic_Deep_Tutorial.CogModelingRNNsTutorial import rnn_utils

def find(s, ch):
  """Find index of character within string."""
  return [i for i, ltr in enumerate(s) if ltr == ch]


def get_rat_bandit_datasets() -> List[rnn_utils.DatasetRNN]:
  """Downloads and packages rat two-armed bandit datasets.

  Dataset is from the following paper:
  From predictive models to cognitive models: Separable behavioral processes
  underlying reward learning in the rat. Miller, Botvinick, and Brody,
  bioRxiv, 2018

  Dataset is available from Figshare at the following link:
  https://figshare.com/articles/dataset/From_predictive_models_to_cognitive_models_Separable_behavioral_processes_underlying_reward_learning_in_the_rat/20449356

  Returns:
    A list of DatasetRNN objects. One element per rat.
    In each of these, each session will be an episode, padded with -1s
    to match length. "ys" will be the choices on each trial
    (left=0, right=1) "xs" will be the choice and reward (0 or 1) from
    the previous trial. Invalid xs and ys will be -1
  """

  # Download the file to the current folder
  url = 'https://figshare.com/ndownloader/files/40442660'
  data_path = 'tab_dataset.json'
  urllib.request.urlretrieve(url, data_path)

  # Load dataset into memory
  with open(data_path, 'r') as f:
    dataset = json.load(f)

  # Clean up after ourselves by removing the downloaded file
  os.remove(data_path)

  # "dataset" will be a list in which each element is a dict. Each of these
  # dicts holds data from a single rat.

  # Each iteration of the loop processes data from one rat, converting the dict
  # into inputs (xs) and targets (ys) for training a neural network, packaging
  # these into a DatasetRNN object, and appending this to dataset_list
  n_rats = len(dataset)
  dataset_list = []
  for rat_i in range(n_rats):
    ratdata = dataset[rat_i]
    # "sides" is a list of characters in which each character specifies the
    # choice made on a trial. 'r' for right, 'l' for left, 'v' for a violation
    # Here, we'll code left choices as 0s, right choices as 1s, and will flag
    # violations for later removal
    sides = ratdata['sides']
    n_trials = len(sides)
    rights = find(sides, 'r')
    choices = np.zeros(n_trials)
    choices[rights] = 1
    vs = find(sides, 'v')
    viols = np.zeros(n_trials, dtype=bool)
    viols[vs] = True

    # Free will be 0 and forced will be 1
    # "trial_types" is a list of characters, each giving the type of a trial.
    # 'f' for free-choice, 'l' for instructed-left, 'r' for instructed-right
    free = find(ratdata['trial_types'], 'f')
    instructed_choice = np.ones(n_trials)
    instructed_choice[free] = 0

    # "rewards" is a list of 1s (rewarded trials) and 0s (unrewarded trials)
    rewards = np.array(ratdata['rewards'])

    # "new_sess" is a list of 1s (trials that are the first of a new session)
    # and 0s (all other trials)
    new_sess = np.array(ratdata['new_sess'])
    n_sess = np.sum(new_sess)
    sess_starts = np.nonzero(np.concatenate((new_sess, [1])))[0]
    # We will pad each session so they all have length of the longest session
    max_session_length = np.max(np.diff(sess_starts, axis=0))

    # Instantiate blank matrices for rewards and choices.
    # size (n_trials, n_sessions, 1)
    rewards_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    choices_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    instructed_by_session = -1 * np.ones((max_session_length, n_sess, 1))

    # Each iteration processes one session
    for sess_i in np.arange(n_sess):
      # Get the choices, rewards, viols, and instructed for just this session
      sess_start = sess_starts[sess_i]
      sess_end = sess_starts[sess_i + 1]
      viols_sess = viols[sess_start:sess_end]
      rewards_sess = rewards[sess_start:sess_end]
      choices_sess = choices[sess_start:sess_end]
      instructed_choice_sess = instructed_choice[sess_start:sess_end]

      # Remove violation trials
      rewards_sess = np.delete(rewards_sess, viols_sess)
      choices_sess = np.delete(choices_sess, viols_sess)
      instructed_choice_sess = np.delete(instructed_choice_sess, viols_sess)
      sess_length_noviols = len(choices_sess)

      # Add them to the matrices
      rewards_by_session[0:sess_length_noviols, sess_i, 0] = rewards_sess
      choices_by_session[0:sess_length_noviols, sess_i, 0] = choices_sess
      instructed_by_session[0:sess_length_noviols, sess_i, 0] = (
          instructed_choice_sess
      )

    # Define neural network inputs:
    # for each trial the choice and reward from the previous trial.
    choice_and_reward = np.concatenate(
        (choices_by_session, rewards_by_session), axis=2
    )
    xs = np.concatenate(
        (0. * np.ones((1, n_sess, 2)), choice_and_reward), axis=0
    )  # First trial's input will arbitrarily always be 0

    # Define neural network targets:
    # choices on all free-choice trial, -1 on all instructed trials
    free_choices = choices_by_session
    free_choices[instructed_by_session == 1] = -1
    # Add a dummy target at the end -- last step has input but no target
    ys = np.concatenate((free_choices, -1*np.ones((1, n_sess, 1))), axis=0)

    # Pack into a DatasetRNN object and append to the list
    dataset_rat = rnn_utils.DatasetRNN(ys=ys, xs=xs)
    dataset_list.append(dataset_rat)

  return dataset_list
