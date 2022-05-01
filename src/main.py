import os.path

import numpy as np
from keras.models import load_model
from classifier import AlexnetClassifier
from utils import *
from detector import *
from test_accuracy import *
from tuning_hyperparam import *
from configparser import ConfigParser

config = ConfigParser()
config.read('../config.ini')
# prepare data
image_size = config.get('DATA','IMAGE_SIZE')
image_channel = config.get('DATA', 'IMAGE_CHANNEL')

# origin data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_preprocessing_mnist(float(config.get('DATA','validationRatio')))

advDataFolder = config.get('DATA', 'advDataFolderPath')
adv_data = config.get('DATA', 'advData')
advTrueLabel = config.get('DATA','advTrueLabel')
classifierPath = config.get('DATA','targetClassifierPath')


classifier = load_model(classifierPath)
_, acc_non_clean = classifier.evaluate(adv_data, advTrueLabel)

defender = DefenderModel(image_size, image_channel)

adv_clean = []
for i in range(len(adv_data)):
    image = get_output_image(adv_data[i], defender)
    adv_clean.append(image)
adv_clean = np.array(adv_clean)
_, acc_clean = classifier.evaluate(adv_clean, advTrueLabel)

result = {
        "number of adv": len(adv_data),
        "adv non clean accuracy": acc_non_clean,
        "adv clean accuracy": acc_clean,
}
print(result)