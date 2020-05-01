import numpy as np
from fcnn_model_manager import FCNNModelManager
from model_configs import filename, class_labels, input_shape, test_size, model_parameters

model_manager = FCNNModelManager(filename=filename, class_labels=class_labels, input_shape=input_shape,
                                 test_size=test_size)

model_names = ('a','b', 'c', 'd',)
models = model_manager.get_trained_models(model_parameters, model_names)
# average_score = model.predict(model_manager.X_test)
# predictions = [np.argmax(item) for item in average_score]
# class_labels = [np.argmax(item) for item in model_manager.y_test]
#
# # Calculating categorical accuracy taking label having highest probability
# accuracy = [(x == y) for x, y in zip(predictions, class_labels)]
# accuracy = round(np.mean(accuracy) * 100, 2)
# print(accuracy)
results = model_manager.get_ensembled_results(models)
for key, value in results.items():
    print(key, value)
