import pandas as pd
import csv
from requests import get
import json
from datetime import datetime, timedelta, date
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr
import csv
import re
import pandas as pd
import argparse
import collections
import json
import glob
import math
import os
import requests
import string
import sys
import time
import xml
import random

class Recommender(object):
    def __init__(self, training_set, test_set):
        if isinstance(training_set, str):
            self.training_set = pd.read_csv(training_set)
        else:
            self.training_set = training_set.copy()
        if isinstance(test_set, str):
            self.test_set = pd.read_csv(test_set)
        else:
            self.test_set = test_set.copy()

    def train_user_euclidean(self, data_set, userId):
        weights = {}
