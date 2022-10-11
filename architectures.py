"""Commonly used neural network architectures."""
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import initializers


def create_architecture(params):
	elif (params['architecture'] == 'pretrained_resnet'):
		net = PretrainedResNet50(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"],
			n_classes=params['n_classes'])
	elif (params['architecture'] == 'pretrained_inception'):
		net = PretrainedInceptionv3(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"],
			n_classes=params['n_classes'])
	else:
		raise NotImplementedError(
			"need to implement other architectures")
	return net

class PretrainedResNet50(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False, n_classes=1):
		super(PretrainedResNet50, self).__init__()
		self.embedding_dim = embedding_dim

		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.resenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))

		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(n_classes,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x


class PretrainedInceptionv3(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False, n_classes=1):
		super(PretrainedInceptionv3, self).__init__()
		self.embedding_dim = embedding_dim

		self.inception = InceptionV3(include_top=False,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.inception.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))

		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(n_classes,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.inception(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x
