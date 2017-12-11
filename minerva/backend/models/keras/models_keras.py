# from keras.models import load_model

from minerva.backend.base import BaseTransformer


class BasicKerasClassifier(BaseTransformer):
	def __init__(self, architecture_config, training_config, callbacks_config):
		self.architecture_config = architecture_config
		self.training_config = training_config
		self.callbacks_config = callbacks_config

		self._build_model = None
		self._build_optimizer = None
		self._build_loss = None

		self.callbacks = self._create_callbacks(**self.callbacks_config)
		self.model = self._compile_model(**self.architecture_config)

	def fit(self, datagen, validation_datagen):
		train_flow, train_steps = datagen
		valid_flow, valid_steps = validation_datagen
		self.model.fit_generator(train_flow,
		                         steps_per_epoch=train_steps,
		                         validation_data=valid_flow,
		                         validation_steps=valid_steps,
		                         callbacks=self.callbacks,
		                         verbose=1,
		                         **self.training_config)
		return self

	def transform(self, datagen, validation_datagen=None):
		test_flow, test_steps = datagen
		predictions = self.model.predict_generator(test_flow, test_steps, verbose=1)
		return {'prediction_proba': predictions}

	def reset(self):
		self.model = self._build_model(**self.architecture_config)

	def _compile_model(self, model_params, optimizer_params):
		model = self._build_model(**model_params)
		optimizer = self._build_optimizer(**optimizer_params)
		loss = self._build_loss()
		model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
		return model

	def _create_callbacks(self, **kwargs):
		return NotImplementedError

	def save(self, filepath):
		self.model.save(filepath)

	def load(self, filepath):
		self.model = load_model(filepath)
		return self