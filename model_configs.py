filename = 'data/fer2013.csv'
class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
test_size = .1
input_shape = (48, 48, 1)
model_parameters = {
    "batch_size": 64,
    "epochs": 100,
    "verbose": 2,
    "validation_split": 0.1111
}
