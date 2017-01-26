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

import pendulum as pdm

import biorep_etl.data.field_definitions.hreg_redcap_dump as rcd
import biorep_etl.data.parsers as parsers
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
    
    # add the forward relationships
    d = Munch({f_name:f_label for f_name,f_label in zip(field_names, field_labels)})
    
    # add the reverse relationships
    d.update(Munch({f_label:f_name for f_label,f_name in zip(field_labels, field_names)}))
    
    return d

def make_choice_maps(data_dict):
    """Return dict of dicts to translate ``int`` values to their meanings.
    
    Generally, the columns will be those whose data type is categorical.
    Exps: 'prior_protocol_number', 'gender', etc.
    
    Args:
        data_dict (pandas.DataFrame): Loaded data_dict object.
        
    Returns:
        Munch: dict of dicts -> key1 = column_names, key2 = ``int``, value = choice text.
    """
    # remove any column_names that have "field type" == 'calc'
    choice_strings = data_dict[data_dict['Field Type'] != 'calc']
    
    # This is all items in the choice defintions column, that are not null, indexed by the data column they pertain to.
    choice_strings = choice_strings[choice_strings['Choices, Calculations, OR Slider Labels'].notnull()]['Choices, Calculations, OR Slider Labels']
    
    
    # Add first level of tree (keys=col_names and vals=Munch())
    maps = Munch({col:Munch() for col in choice_strings.index.values})
    
    # Create parser for choice strings
    choices = parsers.build_choices_parser()
    
    # add second level of tree "maps.col_name" (keys="choice_integer" and vals="choice text")
    for col in maps.keys():
        parsed_choices = choices.parseString(choice_strings[col]).asList()
        maps[col].update({int_:txt_ for int_,txt_ in parsed_choices})
    
    return maps
    
    

def load_data_dict(data_dict_):
    """Load data dict into df."""
    return pd.read_csv(data_dict_, index_col=0)


def infer_crude_dtypes(data_dict, data_path):
    """Use data_dict to infer correct data types for each data column.
    
    Args:
        data_dict (pandas.DataFrame): Loaded data_dict object.
        data_path (Path | str): Location of redcap data dump.
    
    Returns:
        Munch: ``keys in ['','']``
    """
    # Define mapping from redcap type to np.dtype
    rc2np = Munch()
    rc2np.radio = np.float64
    rc2np.text = np.object
    rc2np.checkbox = np.float64
    rc2np.dropdown = np.float64
    rc2np.yesno = np.float64
    rc2np.descriptive = np.object
    rc2np.calc = np.float64
    # this is a special case for certain cols NOT present in the data_dict (see var use_np_objs below)
    rc2np.category = np.object
    
    # Get list of data_col_names
    try:
        data_col_names = data_path.open().readline().strip('\n').split(',')
    except AttributeError:
        data_col_names = Path(data_path).open().readline().strip('\n').split(',')
        
    # Get redcap dtypes for each row of data_dict
    redcap_types = data_dict['Field Type']
    
    # Use redcap_types + data_col_names to associate a reasonible dtype
    col_redcap_types = Munch()
    
    ## distribute redcap_type to each col
    for col in data_col_names:
        try:
            col_redcap_types[col] = redcap_types[col.split('___')[0]]
        except KeyError as exc:
            use_category = {'redcap_event_name',
                            'registration_visit_complete',
                            'baseline_and_follow_up_complete',
                            'surgeries_complete',
                            'hospitalizations_complete'}
            
            missing_key = exc.args[0]
            if missing_key in use_category:
                col_redcap_types[col] = 'category'
            else:
                raise exc
    
    # Set up return dict
    crude_dtypes = Munch()
    crude_dtypes.redcap_dtypes = col_redcap_types
    crude_dtypes.numpy_dtypes = {col_name:rc2np[val] for col_name, val in col_redcap_types.items()}
    
    return crude_dtypes

def recast_advanced_dtypes(data, data_dict, crude_dtypes):
    """Convert dataframe columns to more useful dtypes.
    
    This includes using things like ``pandas.Categorical`` and ``pendulum.Pendulum`` types
    rather that defaulting to ``str`` or ``float`` primatives.
    
    Args:
        data (pandas.DataFrame): Loaded and crudly re-typed redcap data dump.
        data_dict (pandas.DataFrame): Loaded data_dict object.
        crude_dtypes (Munch): Output from ``infer_crude_dtypes()``.
        
    Returns:
        None: Modifies ``data`` in-place.
    """
    recast_as = Munch()
    
    # Collect Columns to re-cast
    ## Category Columns
    redcap_cats = {"radio",
                   "checkbox",
                   "dropdown",
                   "yesno",
                   "category",}

    recast_as.category = [col for col,val in crude_dtypes.redcap_dtypes.items() if val in redcap_cats]
    
    ## Date Columns
    dt_vals = {'date_mdy', 'date_dmy'}
    recast_as.date =  list(data_dict[data_dict['Text Validation Type OR Show Slider Number'].isin(dt_vals)].index.values)
    
    
    # Do the re-casts
    ## Categories
    [cast_column_as_category(df=data, col=c) for c in recast_as.category]
    
    ## Dates
    [cast_column_as_date(df=data, col=c) for c in recast_as.date]
    

def load_redcap_dump(data_, data_dict):
    """Return loaded, recode, and validated dump table.
    
    Args:
        data_ (Path): Location of redcap data dump.
        data_dict (pandas.DataFrame): Loaded data_dict object.
        
    Returns:
        pandas.DataFrame
    """
    crude_dtypes = infer_crude_dtypes(data_dict=data_dict, data_path=data_)
    
    data = pd.read_csv(data_, dtype=crude_dtypes.numpy_dtypes, index_col=None)
    
    recast_advanced_dtypes(data=data, data_dict=data_dict, crude_dtypes=crude_dtypes)
    
    
    return data.set_index(['subid','redcap_event_name']).sort_index(level='subid')


def load_all_and_labels(data_, data_dict_):
    """Return Munch obj containing the loaded and verified data dump and the field_map."""
    loaded = Munch()
    
    data_dict = load_data_dict(data_dict_=data_dict_)
    required = rcd.req_cols(dd=data_dict)
    
    loaded.data = load_redcap_dump(data_=data_)
    
    
    loaded.field_map = make_field_map(data_dict=data_dict)
    
    

    
    return loaded
    
    



########################## True Column Recoding Functions ##########################

def cast_func_ignore_nulls(x, f):
    """Meant to be past to ``df.column_name.apply()``.

    Args:
        x (??): item in a ``pandas.Series``
        func (function): function that casts ``x`` as some type.

    Returns:
        f(x) | x: re-cast value of ``x`` or original ``x`` if ``f`` fails.
    """
    if pd.notnull(x):
        return f(x)
    else:
        return x



def cast_column_as_date(df, col):
    """Peform in-place re-casting of ``df[col]`` to ``pendulum.Pendulum``.
    
    Ignores nulls.
    
    Args:
        df (pandas.DataFrame): a dataframe.
        col (str): column name in ``df`` to be re-cast.
    """
    df.loc[:,col] = df[col].apply(cast_func_ignore_nulls, f=pdm.parse)


def cast_column_as_category(df, col):
    """Peform in-place re-casting of ``df[col]`` to ``pendulum.Pendulum``.
    
    Args:
        df (pandas.DataFrame): a dataframe.
        col (str): column name in ``df`` to be re-cast.
    """
    df[col] = df[col].astype('category')
