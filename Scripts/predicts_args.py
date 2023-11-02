import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import pickle
from UrlFeaturizer import UrlFeaturizer

# Load the LabelEncoder and StandardScaler
encoder = LabelEncoder()
encoder.classes_ = np.load('lblenc.npy', allow_pickle=True)
scaler = pickle.load(open('scaler.sav', 'rb'))

# Load the model
model = load_model("best_model.h5")

def main():
    parser = argparse.ArgumentParser(description="URL Classification")
    parser.add_argument("-i", "--url", required=True, help="URL to classify")

    args = parser.parse_args()
    url = args.url

    # Featurize the URL
    url_featurizer = UrlFeaturizer(url)
    features = url_featurizer.run()

    # The order of the features should be the same as the training data
    order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
            'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
            'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
            'sscr', 'urlIsLive', 'urlLength']

    # Prepare the features for prediction
    test = [features[i] for i in order]
    test = pd.DataFrame(test).replace(True, 1).replace(False, 0).to_numpy().reshape(1, -1)

    # Standardize the features
    test = scaler.transform(test)

    # Reshape the input data to fit the model input shape
    test = test.reshape(test.shape[0], test.shape[1], 1)

    # Make the prediction
    predicted = np.argmax(model.predict(test), axis=1)

    # Convert the numerical prediction to class label
    predicted_class = encoder.inverse_transform(predicted)[0]
    print(f"Predicted class for URL '{url}': {predicted_class}")

if __name__ == "__main__":
    main()
