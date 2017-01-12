#!/usr/bin/env python
"""Provide code to validate loaded data tables."""

# Imports
import os
from collections import defaultdict
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from munch import Munch

import biorep_etl.errors as e

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"


# Constants
# dtypes = defaultdict(Munch)
# dtypes['redcap_dump'].



# Functions
