import argparse
import pandas as pd 
import functools
import waterbirds.data_builder as wb
import tensorflow as tf
import numpy as np
from causallearn.utils.KCI.KCI import KCI_CInd

# NOTE: the original causallearn code was modified to take epsilon as an argument. 
# in the original code, its value is hardcoded. 

def get_waterbirds_high_dim_x(random_seed, base_dir, pixel=128, 
	weighted=False, batch_size=1000):
	"""Function to get the data."""
	experiment_directory = (
			f"{base_dir}/experiment_data/rs{random_seed}")
	train_data = pd.read_csv(
			f'{experiment_directory}/train.txt'
			).values.tolist()
	train_data = [
			tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]
			
	map_to_image_label_given_pixel = functools.partial(wb.map_to_image_label,
					pixel=pixel, weighted=weighted)

	train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
	train_dataset = train_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	train_dataset = train_dataset.batch(batch_size,
			drop_remainder=False).repeat(1)
		
	tr_data = []
	for batch_id, examples in enumerate(train_dataset):
			print(batch_id)
			x, labels_weights = examples
			x = x.numpy()
			x = x.reshape(x.shape[0], pixel*pixel*3)
			pix_cols = [f'x{i}' for i in range(x.shape[1])]
			labels = labels_weights['labels'].numpy()
			label_cols = [f'y{i}' for i in range(labels.shape[1])]
			x = pd.DataFrame(np.hstack([x, labels]), 
											columns = pix_cols + label_cols)
			tr_data.append(x)

	tr_data = pd.concat(tr_data)
	return tr_data
		
def get_high_dim_x(experiment_name, random_seed, base_dir,
	pixel=128, weighted=False, batch_size=1000):
	if experiment_name == "waterbirds":
		return	get_waterbirds_high_dim_x(
			random_seed=random_seed, 
			base_dir=base_dir, 
			pixel=pixel, 
			weighted=weighted, 
			batch_size=batch_size)
	else: 
		raise NotImplementedError("not yet!")

def get_y_waterbirds(random_seed, base_dir):
	experiment_directory = (
		f"{base_dir}/experiment_data/rs{random_seed}")

	tr_data = pd.read_csv(f'{experiment_directory}/train.txt')
	tr_data
	tr_data = tr_data['0'].str.split(",", expand=True)
	D = tr_data.shape[1]-4
	tr_data.columns = ['bird_img', 'bird_seg', 'back_img', 'noise_img'] + \
			[f'y{i}' for i in range(D)]
	tr_data.drop(['bird_img', 'bird_seg', 'back_img', 'noise_img'], axis=1, inplace=True)
	for i in range(D):
			tr_data[f'y{i}'] = tr_data[f'y{i}'].astype(np.float32)

	return tr_data 

def get_y(experiment_name, random_seed, base_dir):
	if experiment_name == 'waterbirds':
		return get_y_waterbirds(random_seed=random_seed, 
			base_dir=base_dir)
	else: 
		raise NotImplementedError("not yet!")

def y_independence_test(random_seed, v_dim, base_dir, experiment_name):
	df = get_y(experiment_name=experiment_name, 
		random_seed=random_seed, 
		base_dir=base_dir
		)
	D = v_dim + 1

	var_variance = df.var(axis=0)
	var_variance = (var_variance == 0.0).tolist()
	if sum(var_variance) > 0.0:
		drop_vars = [i for i, i_true in enumerate(var_variance) if i_true]
		print(f'the following variables had 0 variance {df.columns[drop_vars]}')
		non_zero_var_labels = list(set(range(D)) - set(drop_vars))
	else: 
		non_zero_var_labels = [i for i in range(D)]

	p_values = []
	for i in range(1, D):			
		if i in non_zero_var_labels:
			kci_obj = KCI_CInd(kernelX='Gaussian', 
								 kernelY='Gaussian', 
								 kernelZ='Gaussian',
								 est_width='median', 
								 epsilon=1e-3)
		
			p, _ = kci_obj.compute_pvalue(
				data_x=df.values[:, 0][:, np.newaxis],
				data_y=df.values[:, i][:, np.newaxis], 
				data_z = df.values[:,[j for j in non_zero_var_labels if ((j!=0) & (j!=i))]]
				)
		else: 
			p = 0.0
		p_values.append(p)
	return p_values

def x_independence_test(random_seed, pixel, batch_size,
	 model_name, xv_mode, v_dim, base_dir, x_mode, experiment_name):
	if x_mode == 'sufficient_stat': 
		df = pd.read_csv((f'{base_dir}/final_models/opt_pred_rs{random_seed}_{model_name}_{xv_mode}'
			f'_pix{pixel}_bs{batch_size}_vdim{v_dim}.csv'))
		df.columns = df.columns.str.replace("pred", "x")
		df.drop(['model', 'dist'], inplace=True, axis=1)
	else: 
		df = get_high_dim_x(
			experiment_name=experiment_name, 
			random_seed=random_seed, 
			base_dir=base_dir, 
			pixel=128, 
			weighted=False, 
			batch_size=1000)


	D = v_dim + 1
	y_cols = [i for i in range(df.shape[1] - D, df.shape[1])]
	x_cols = list(set(range(df.shape[1])) - set(y_cols))

	# if any of the variables have 0 variance, the test fails
	# need to remove them first. 
	var_variance = df.var(axis=0)
	
	var_variance = (var_variance == 0.0).tolist()
	if sum(var_variance) > 0.0:
		drop_vars = [i for i, i_true in enumerate(var_variance) if i_true]
		print(f'the following variables had 0 variance {df.columns[drop_vars]}')
		drop_y_vars = list(set(y_cols) & set(drop_vars))
		y_cols_reduced = list(set(y_cols) - set(drop_vars))
		x_cols = list(set(x_cols) - set(drop_vars))
	else: 
		y_cols_reduced = y_cols

	p_values = []
	for i in y_cols[1:]:
		if i in y_cols_reduced:
			kci_obj = KCI_CInd(kernelX='Gaussian', 
												 kernelY='Gaussian', 
												 kernelZ='Gaussian',
												 est_width='median', 
												 epsilon=1e-3)
			
			p, _ = kci_obj.compute_pvalue(
				data_x=df.values[:, x_cols],
				data_y=df.values[:, i][:, np.newaxis], 
				data_z = df.values[:,[j for j in y_cols_reduced if (j!=i)]]
					)
		else: 
			p = 0.0
		p_values.append(p)

	return p_values

def main(seed_list, x_mode, test_pval, experiment_name, pixel, 
	batch_size, model_name, xv_mode, v_dim, base_dir, **args): 
	seed_list = [int(i) for i in seed_list.split(",")]

	results = []
	for random_seed in seed_list:
		print(f'======= Random seed :{random_seed} ========')
		x_pvals = x_independence_test(
			random_seed = random_seed, 
			pixel = pixel, 
			batch_size = batch_size, 
			model_name = model_name, 
			xv_mode = xv_mode, 
			v_dim = v_dim, 
			base_dir = base_dir, 
			x_mode = x_mode, 
			experiment_name=experiment_name
			)


		y_pvals = y_independence_test(
			random_seed = random_seed, 
			v_dim=v_dim, 
			base_dir = base_dir,
			experiment_name=experiment_name
			)

		x_rel_bool = [x_pval <= test_pval for x_pval in x_pvals]
		x_rel = [i + 1 for i, i_true in enumerate(x_rel_bool) if i_true]

		y_rel_bool = [y_pval <= test_pval for y_pval in y_pvals]
		y_rel = [i + 1 for i, i_true in enumerate(y_rel_bool) if i_true]

		rel_aux = list(set(x_rel) & set(y_rel))

		rs_results = [random_seed, x_mode] + x_pvals + y_pvals + [rel_aux]
		results.append(rs_results)
	

	results = pd.DataFrame(results)
	results.columns = ['random_seed', 'x_mode'] + \
		[f'x_ci_y{i}' for i in range(1, len(x_pvals) + 1)] + \
		[f'y0_ci_y{i}' for i in range(1, len(y_pvals) + 1)] + \
		['relevant_aux_labs']
	
	print(results[['random_seed', 'x_mode', 'relevant_aux_labs']].head(10))


	results.to_csv(
		(f'{base_dir}/final_models/indep_tests_{x_mode}'
		f'_pix{pixel}_bs{batch_size}.csv'),
		index=False)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--experiment_name', '-experiment_name',
		help="which dataset?",
		choices=['waterbirds', 'chexpert', 'dr'], 
		type=str)

	parser.add_argument('--base_dir', '-base_dir',
		help="Base directory where the final model will be saved",
		type=str)


	parser.add_argument('--random_seed', '-random_seed',
		help="Random seed for which we want to get predictions",
		default=-1,
		type=int)

	parser.add_argument('--seed_list', '-seed_list',
		help=("Comma separated list of seeds to get predictions for. "
				"overrides random seed"),
		type=str)

	parser.add_argument('--batch_size', '-batch_size',
		help="batch size",
		type=int)

	parser.add_argument('--pixel', '-pixel',
		help="pixels",
		type=int)

	parser.add_argument('--v_dim', '-v_dim',
		help="number of additional dimensions",
		type=int)

	parser.add_argument('--model_name', '-model_name',
		default='first_step',
		help="Which model to predict for",
		type=str)

	parser.add_argument('--xv_mode', '-xv_mode',
		default='classic',
		choices=['classic', 'two_step'],
		help=("which cross validation algorithm do you want to get preds for"),
		type=str)

	parser.add_argument('--x_mode', '-x_mode',
		default='sufficient_stat',
		choices=['sufficient_stat', 'high_dim'],
		help=("use high dim raw x or reduce using sufficient statistic?"),
		type=str)

	parser.add_argument('--test_pval', '-test_pval',
		default=0.001,
		help=("what is on the pval for the independence test?"),
		type=float)

	args = vars(parser.parse_args())
	main(**args)