import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1" # ARM chips on mac don't yet support certain features. make sure this is the VERY FIRST command to run.

import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

AirPassengers = AirPassengersPanel.to_csv("AIrPassengersPanel.csv",index=True)
AirPassengersStatic = AirPassengersStatic.to_csv("AIrPassengersStatic.csv",index=True)