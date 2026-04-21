import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import t


def main():
   # Load data
   with  open('data/Horns_rev1.pkl', 'rb') as file:
     raw_data = pickle.load(file)

   # Preprocess
   raw_data['ws_x'] = raw_data.apply(lambda x: x.ws * np.cos(np.pi * x.wd / 180), axis=1)
   raw_data['ws_y'] = raw_data.apply(lambda x: x.ws * np.sin(np.pi * x.wd / 180), axis=1)

   scaler = MinMaxScaler(feature_range=(0, 1))
   data = pd.DataFrame(scaler.fit_transform(raw_data), columns=raw_data.columns)

   # Split into train-validation-test

   # Define model
   
   # Run training


if __name__ == "__main__":
   main()
 