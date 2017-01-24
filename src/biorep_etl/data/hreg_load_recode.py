#!/usr/bin/env python
"""Provide logic to import, parse, and recode project data into usable structures for analysis."""

# Imports
import os
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from munch import Munch

import engarde.decorators as ed
import engarde.checks as ck


import biorep_etl.data.field_definitions.hreg_redcap_dump as rcd
import biorep_etl.errors as e

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"


# functions
def make_field_map(data_dict):
    """Return Dict to translate field names and field labels."""
    d = Munch()
    field_names = data_dict.reset_index()['Variable / Field Name'].values
    field_labels = data_dict['Field Label'].values
    
    d = Munch({k:v for k,v in zip(field_names, field_labels)})
    d.update(Munch({d:l for d,l in zip(field_labels, field_names)}))
    
    return d

def load_data_dict(data_dict_):
    """Load data dict into df."""
    return pd.read_csv(data_dict_, index_col=0)



def load_redcap_dump(data_):
    """Return loaded, recode, and validated dump table."""
    # dtypes = {'samplenumber': np.object,
    #           'mrn': np.object,
    #           'biorepidnumber': np.object,}
    
    return pd.read_csv(data_, dtype=None, index_col=None)
    # return pd.read_csv(data_, dtype=None, index_col=0, low_memory=False)
    


def init_load_data_and_labels(data_, data_dict_):
    """Return Munch obj containing the loaded and verified data dump and the field_map."""
    loaded = Munch()
    
    data_dict = load_data_dict(data_dict_=data_dict_)
    required = rcd.req_cols(dd=data_dict)
    
    loaded.data = load_redcap_dump(data_=data_)
    
    
    loaded.field_map = make_field_map(data_dict=data_dict)
    
    

    
    return loaded
    
    






