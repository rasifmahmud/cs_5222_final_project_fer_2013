import os
from itertools import combinations

import numpy as np
from keras.layers import Dense, Activation, Dropout, GlobalAveragePooling2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.model_selection import train_test_split


class FCNNModelManager:
    SAVED_MODEL_PATH = "./saved_models"
    MODEL_NAMES = ["a", "b", "c", "d"]

    def __init__(self, filename, class_labels, input_shape=(48, 48, 1), test_size=.1):
        self.filename = filename
        self.class_labels = class_labels
        self.number_of_classes = len(class_labels)
        self.input_shape = input_shape
        self.test_size = test_size
        self.preprocess_data()

    def preprocess_data(self):
        Y = []
        X = []
        first = True
        for line in open(self.filename):
            if first:
                first = False
            else:
                row = line.split(',')
                Y.append(int(row[0]))
                X.append([int(p) for p in row[1].split()])

        X, Y = np.array(X) / 255.0, np.array(Y)

        N, D = X.shape
        X = X.reshape(N, 48, 48, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size,
                                                                                random_state=0)
        self.y_train = (np.arange(self.number_of_classes) == self.y_train[:, None]).astype(np.float32)
        self.y_test = (np.arange(self.number_of_classes) == self.y_test[:, None]).astype(np.float32)

    def generate_model_combinations(self):
        model_dict = {}
        for i in range(1, len(self.MODEL_NAMES) + 1):
            model_combinations = combinations(self.MODEL_NAMES, i)
            for combination in model_combinations:
                model_dict[combination] = None
        return model_dict

    @staticmethod
    def get_saved_model_file_names(model_name, model_parameters):
        json_file_name = "model_{}_".format(model_name)
        h5_file_name = "model_{}_".format(model_name)
        for key, value in model_parameters.items():
            json_file_name += str(key) + "_" + str(value) + "_"
            h5_file_name += str(key) + "_" + str(value) + "_"
        json_file_name = json_file_name.replace(".", "_")
        h5_file_name = h5_file_name.replace(".", "_")
        json_file_name = json_file_name[:-1] + ".json"
        h5_file_name = h5_file_name[:-1] + ".h5"
        return json_file_name, h5_file_name

    def get_trained_models(self, model_parameters, model_names):
        models = dict()
        for model_name in model_names:
            json_file_name, h5_file_name = self.get_saved_model_file_names(model_name, model_parameters)
            if os.path.exists(os.path.join(self.SAVED_MODEL_PATH, json_file_name)) and os.path.exists(
                    os.path.join(self.SAVED_MODEL_PATH, h5_file_name)):
                model = self.get_saved_model(json_file_name, h5_file_name)
            else:
                model = self.train_model(model_name, model_parameters)
            model.name = model_name
            models[model.name] = model
        return models

    def get_saved_model(self, json_file_name, h5_file_name):
        json_file = open(os.path.join(self.SAVED_MODEL_PATH, json_file_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights from h5 file
        model.load_weights(os.path.join(self.SAVED_MODEL_PATH, h5_file_name))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
        return model

    def get_baseline_model(self):
        # Initialising the CNN
        model = Sequential()

        # 1 - Convolution
        model.add(Conv2D(96, (3, 3), border_mode='same', input_shape=self.input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(96, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(96, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Dropout(0.5))

        # 2nd Convolution layer
        model.add(Conv2D(192, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(192, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(192, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Dropout(0.5))

        model.add(Conv2D(192, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        return model

    def get_model_a(self):
        model = self.get_baseline_model()

        model.add(Conv2D(192, (1, 1), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(7, (1, 1), border_mode='same'))
        model.add(Activation('relu'))

        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(self.number_of_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

        return model

    def get_model_b(self):
        model = self.get_baseline_model()

        model.add(Conv2D(192, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Dropout(0.5))

        model.add(Conv2D(256, (1, 1), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(self.number_of_classes, (1, 1), border_mode='same'))
        model.add(Activation('relu'))

        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(self.number_of_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

        return model

    def get_model_c(self):
        model = self.get_baseline_model()

        model.add(Conv2D(192, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Dropout(0.5))

        model.add(Conv2D(256, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (1, 1), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(self.number_of_classes, (1, 1), border_mode='same'))
        model.add(Activation('relu'))

        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(self.number_of_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

        return model

    def get_model_d(self):
        model = self.get_baseline_model()

        model.add(Conv2D(192, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Dropout(0.5))

        model.add(Conv2D(256, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Dropout(0.5))

        model.add(Conv2D(512, (1, 1), border_mode='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(self.number_of_classes, (1, 1), border_mode='same'))
        model.add(Activation('relu'))

        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(self.number_of_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

        return model

    def save_model_in_file(self, model, model_name, model_parameters):
        model_json = model.to_json()
        json_file_name, h5_file_name = self.get_saved_model_file_names(model_name, model_parameters)
        with open(os.path.join(self.SAVED_MODEL_PATH, json_file_name), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(self.SAVED_MODEL_PATH, h5_file_name))

    def train_model(self, model_name, model_parameters):
        model = getattr(self, "get_model_{}".format(model_name))()
        model.fit(self.X_train, self.y_train,
                  batch_size=model_parameters['batch_size'],
                  epochs=model_parameters['epochs'],
                  verbose=model_parameters['verbose'],
                  validation_split=model_parameters['validation_split'])
        self.save_model_in_file(model, model_name, model_parameters)
        return model

    def get_ensembled_results(self, models):
        model_combinations = self.generate_model_combinations()
        for model_combination in model_combinations:
            scores = list()
            for model_name in model_combination:
                model = models[model_name]
                score = model.predict(self.X_test)
                scores.append(score)
            average_score = sum(scores) / len(scores)
            predictions = [np.argmax(item) for item in average_score]
            class_labels = [np.argmax(item) for item in self.y_test]

            # Calculating categorical accuracy taking label having highest probability
            accuracy = [(x == y) for x, y in zip(predictions, class_labels)]
            model_combinations[model_combination] = round(np.mean(accuracy) * 100, 2)
        return model_combinations
