# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates config dictionaries for different experiments and models waterbirds"""
import os
import functools
from pathlib import Path
from random import sample
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing
import tqdm
from copy import deepcopy
from shared import weighting as wt

tf.autograph.set_verbosity(0)

import chexpert_support_device.data_builder as chx
import waterbirds.data_builder as wb
import dr.data_builder as dr
import shared.train_utils as utils
from shared import evaluation

def get_last_saved_model(estimator_dir):
	subdirs = [x for x in Path(estimator_dir).iterdir()
		if x.is_dir() and 'temp' not in str(x)]
	try:
		latest_model_dir = str(sorted(subdirs)[-1])
		loaded = tf.saved_model.load(latest_model_dir)
		model = loaded.signatures["serving_default"]
	except:
		print(estimator_dir)
	return model

def get_optimal_sigma_for_run(config, valid_dataset, base_dir, t1_error, n_permute=100):
	# -- model
	hash_string = utils.config_hasher(config)
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')
	model = get_last_saved_model(hash_dir)

	# ---compute hsic over folds
	z_pred_list = []
	labels_list = []
	sample_weights_list = []
	for batch_id, examples in enumerate(valid_dataset):
		# print(f'{batch_id} / {n_permute}')
		x, labels_weights = examples
		sample_weights = labels_weights['sample_weights']
		sample_weights_list.append(sample_weights)

		labels = labels_weights['labels']
		labels_list.append(labels)

		zpred = model(tf.convert_to_tensor(x))['embedding']
		z_pred_list.append(zpred)

	zpred = tf.concat(z_pred_list, axis=0)
	labels = tf.concat(labels_list, axis=0)
	sample_weights = tf.concat(sample_weights_list, axis=0)

	hsic_val = evaluation.hsic(
		x=zpred, y=labels[:, 1:],
		sample_weights=sample_weights,
		sigma=config['sigma'])[[0]].numpy()

	perm_hsic_val = []
	for seed in range(n_permute):
		# if seed % 10 ==0:
		# 	print(f'{seed}/{n_permute}')
		labels_p = labels.numpy()
		np.random.RandomState(seed).shuffle(labels_p)
		labels_p = tf.constant(labels_p)

		perm_hsic_val.append(evaluation.hsic(
			x=zpred, y=labels_p[:, 1:],
			sample_weights=sample_weights,
			sigma=config['sigma'])[[0]].numpy())

	perm_hsic_val = np.concatenate(
		perm_hsic_val, axis=0)

	thresh = np.quantile(perm_hsic_val, 1 - t1_error)
	accept_null = hsic_val <= thresh
	print(config['random_seed'], config['sigma'], config['alpha'], hsic_val, thresh, accept_null[0])

	curr_results = pd.DataFrame({
		'random_seed': config['random_seed'],
		'alpha': config['alpha'],
		'sigma': config['sigma'],
		'hsic': hsic_val,
		'significant': accept_null
	}, index=[0])

	perm_vals = pd.DataFrame(perm_hsic_val).transpose()
	perm_vals.columns = [f'hsicp{i}' for i in range(len(perm_hsic_val))]
	curr_results = pd.concat([curr_results, perm_vals], axis=1)
	return curr_results

def get_optimal_sigma(all_config, t1_error, valid_dataset, n_permute,
	num_workers, base_dir):
	all_results = []
	runner_wrapper = functools.partial(get_optimal_sigma_for_run_perm,
		base_dir=base_dir, valid_dataset=valid_dataset, t1_error=t1_error, n_permute=n_permute)
	if num_workers <= 0:
		for cid, config in enumerate(all_config):
			print(cid)
			results = runner_wrapper(config)
			all_results.append(results)
	else:
		pool = multiprocessing.Pool(num_workers)
		for results in tqdm.tqdm(pool.imap_unordered(runner_wrapper, all_config),
			total=len(all_config)):
			all_results.append(results)

	all_results = pd.concat(all_results, axis=0, ignore_index=True)
	return all_results





