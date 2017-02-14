#!/usr/bin/env python
"""Provide helper functions for repetitive things involved in writing things that interact with redcap dumps."""

# Imports

import numpy as np
import pandas as pd

from munch import Munch, munchify

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"


# Functions
def get_dd_index_from_str(dd, string):
    dd = dd.reset_index()
    return dd[dd['Variable / Field Name'] == string].index.values[0]



def table_maker_helper(dd,start,end):
    m = Munch()
    
    m.span = dd.reset_index().iloc[start:end,:]['Variable / Field Name']
    m.dd = dd.iloc[m.span.index.values]
    m.type_counts = m.dd['Field Type'].value_counts()
    m.fields = list(m.dd.index.values)
    m.checkboxes = list(m.dd[m.dd['Field Type'] == 'checkbox'].index.values)
    m.not_checkboxes = [f for f in m.fields if f not in m.checkboxes]
    
    return m