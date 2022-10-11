""" Evaluation metrics """
import tensorflow as tf
import tensorflow_probability as tfp

def compute_loss(labels, logits, embedding, sample_weights, params):
	if params['weighted'] == 'False':
		prediction_loss, hsic_loss = compute_loss_unweighted(labels, logits,
			embedding, params)
	else:
		prediction_loss, hsic_loss = compute_loss_weighted(labels, logits,
			embedding, sample_weights, params)
	return prediction_loss, hsic_loss


def compute_loss_unweighted(labels, logits, embedding, params):
	if params['n_classes'] ==1:
		y_main = tf.expand_dims(labels[:, 0], axis=-1)

		individual_losses = tf.keras.losses.binary_crossentropy(y_main, logits,
			from_logits=True)
	else:
		y_main = tf.one_hot(tf.cast(labels[:, 0], tf.int32), params['n_classes'])
		individual_losses = tf.keras.losses.categorical_crossentropy(y_main,
			logits, from_logits=True)

	unweighted_loss = tf.reduce_mean(individual_losses)
	aux_y = labels[:, 1:]
	if params['alpha'] > 0:
		hsic_loss = hsic(embedding, aux_y, sample_weights=None,
			sigma=params['sigma'])
	else:
		hsic_loss = 0.0
	return unweighted_loss, hsic_loss


def compute_loss_weighted(labels, logits, embedding, sample_weights, params):
	if params['n_classes'] == 1:
		y_main = tf.expand_dims(labels[:, 0], axis=-1)
		individual_losses = tf.keras.losses.binary_crossentropy(y_main, logits,
			from_logits=True)
	else:
		y_main = tf.one_hot(tf.cast(labels[:, 0], tf.int32), params['n_classes'])
		individual_losses = tf.keras.losses.categorical_crossentropy(y_main,
			logits, from_logits=True)

	weighted_loss = sample_weights * individual_losses
	weighted_loss = tf.math.divide_no_nan(
		tf.reduce_sum(weighted_loss),
		tf.reduce_sum(sample_weights)
	)

	aux_y = labels[:, 1:]
	if params['alpha'] > 0:
		hsic_loss = hsic(embedding, aux_y, sample_weights=sample_weights,
			sigma=params['sigma'])
	else:
		hsic_loss = 0.0
	return weighted_loss, hsic_loss


def hsic(x, y, sample_weights, sigma=1.0):
	""" Computes the weighted HSIC between two arbitrary variables x, y"""
	if len(x.shape) == 1:
		x = tf.expand_dims(x, axis=-1)

	if len(y.shape) == 1:
		y = tf.expand_dims(y, axis=-1)

	if sample_weights == None:
		sample_weights = tf.ones((tf.shape(y)[0], 1))

	if len(sample_weights.shape) == 1:
		sample_weights = tf.expand_dims(sample_weights, axis=-1)

	sample_weights_T = tf.transpose(sample_weights)

	kernel_fxx = tfp.math.psd_kernels.ExponentiatedQuadratic(
			amplitude=1.0, length_scale=sigma)

	kernel_xx = kernel_fxx.matrix(x, x)
	kernel_fyy = tfp.math.psd_kernels.ExponentiatedQuadratic(
			amplitude=1.0, length_scale=sigma)
	kernel_yy = kernel_fyy.matrix(y, y)

	N = tf.cast(tf.shape(y)[0], tf.float32)

	# First term
	hsic_1 = tf.math.multiply(kernel_xx, kernel_yy)
	hsic_1 = tf.linalg.matmul(sample_weights_T, hsic_1)
	hsic_1 = tf.linalg.matmul(hsic_1, sample_weights)
	hsic_1 = hsic_1 / (N **2)

	# Second term
	# Note there is a typo in the paper. Authors will update
	W_matrix = tf.linalg.matmul(sample_weights, sample_weights_T)
	hsic_2 = tf.math.multiply(kernel_yy, W_matrix)
	hsic_2 = tf.reduce_sum(hsic_2, keepdims=True) * tf.reduce_sum(kernel_xx, keepdims=True)
	hsic_2 = hsic_2 / (N ** 4)

	# third term
	hsic_3 = tf.linalg.matmul(kernel_yy, sample_weights)
	hsic_3 = tf.math.multiply(
			tf.reduce_sum(kernel_xx, axis=1, keepdims=True), hsic_3)
	hsic_3 = tf.linalg.matmul(sample_weights_T, hsic_3)
	hsic_3 = 2 * hsic_3 / (N ** 3)

	hsic_val = hsic_1 + hsic_2 - hsic_3
	hsic_val = tf.maximum(0.0, hsic_val)
	return hsic_val


def auroc(labels, predictions):
	""" Computes AUROC """
	auc_metric = tf.keras.metrics.AUC(name="auroc")
	auc_metric.reset_states()
	auc_metric.update_state(y_true=labels, y_pred=predictions)
	return auc_metric

def get_hsic_at_sigmas(sigma_list, labels, embedding, sample_weights,
	eager):
	result_dict = {}
	for sigma_val in sigma_list:
		hsic_at_sigma = hsic(embedding, labels[:, 1:], sample_weights=sample_weights,
		 sigma=sigma_val)
		if eager:
			result_dict[f'hsic{sigma_val}'] = hsic_at_sigma.numpy()

		else:
			result_dict[f'hsic{sigma_val}'] = tf.compat.v1.metrics.mean(
				hsic_at_sigma)
	return result_dict


def get_eval_metrics_dict(labels, predictions, sample_weights, params,
	eager=False,
	sigma_list=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]):
	n_classes = params['n_classes']
	del params
	if n_classes ==1:
		y_main = tf.expand_dims(labels[:, 0], axis=-1)
	else:
		y_main = tf.one_hot(tf.cast(labels[:, 0], tf.int32), n_classes)

	eval_metrics_dict = {}
	eval_metrics_dict["auc"] = auroc(
		labels=y_main, predictions=predictions["probabilities"])
	
	hsic_val_dict = get_hsic_at_sigmas(sigma_list, labels,
		predictions['embedding'], sample_weights, eager=eager)

	return {**eval_metrics_dict, **hsic_val_dict}

