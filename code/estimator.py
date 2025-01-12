from gluonts.mx import SimpleFeedForwardEstimator, Trainer



def get_estimator():
    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=dataset.metadata.prediction_length,
        context_length=100, # input length
        trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
    )