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

import biorep_etl.data.parsers as parsers
import biorep_etl.errors as e

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"


class BaseRedCapData(object):
    """Organize the common loading, preparation, and storing of RedCap data dumps."""

    def __init__(self, data_path, data_dict_path):
        """Load, recode, and verify various sets of information related to a RedCap dump.

        Args:
            data_path (Path): Location of redcap csv dump.
            data_dict_path (Path): Location of data_dict csv.
        """


    def load_data_dict(self, data_dict_path):
        """Load data dict into dataframe."""
        return pd.read_csv(data_dict_path, index_col=0)


class RegistryRedCapData(object):
    """Organize the Registry-specific loading, preparation, and storing of RedCap data dumps."""

    def __init__(self, data_path, data_dict_path):
        """Load, recode, and verify various sets of information related to a RedCap dump.

        Args:
            data_path (Path): Location of redcap csv dump.
            data_dict_path (Path): Location of data_dict csv.
        """
        super(RegistryRedCapData, self).__init__()

        self._load_data_dict(data_dict_path=data_dict_path)
        self._make_required_columns()

        self._make_field_map()
        self._make_choice_maps()
        self._make_redcap_validation_table()
        
        self._infer_crude_dtypes(data_path=data_path)
        
        self._load_redcap_dump(data_path=data_path)

    def _make_required_columns(self):
        """Return list of columns that are required."""
        self.required_columns = list(self.data_dict[self.data_dict['Required Field?'] == 'y'].index.values)

    def _load_data_dict(self, data_dict_path):
        """Load data dict into dataframe."""
        self.data_dict = pd.read_csv(data_dict_path, index_col=0)

    def _make_field_map(self):
        """Return Dict to translate field names and field labels."""
        d = Munch()
        field_names = self.data_dict.reset_index()['Variable / Field Name'].values
        field_labels = self.data_dict['Field Label'].values

        # add the forward relationships
        d = Munch({f_name: f_label for f_name, f_label in zip(field_names, field_labels)})

        # add the reverse relationships
        d.update(Munch({f_label: f_name for f_label, f_name in zip(field_labels, field_names)}))

        self.field_map = d

    def _make_choice_maps(self):
        """Return dict of dicts to translate ``int`` values to their meanings.

        Generally, the columns will be those whose data type is categorical.
        Examples: 'prior_protocol_number', 'gender', etc.


        Returns:
            Munch: dict of dicts -> key1 = column_names, key2 = ``int``, value = choice text.
        """
        # remove any column_names that have "field type" == 'calc'
        choice_strings = self.data_dict[self.data_dict['Field Type'] != 'calc']

        # This is all items in the choice definitions column, that are not null, indexed by the data column they pertain to.
        choice_strings = choice_strings[choice_strings['Choices, Calculations, OR Slider Labels'].notnull()][
            'Choices, Calculations, OR Slider Labels']

        # Add first level of tree (keys=col_names and vals=Munch())
        maps = Munch({col: Munch() for col in choice_strings.index.values})

        # Create parser for choice strings
        choices = parsers.build_choices_parser()

        # add second level of tree "maps.col_name" (keys="choice_integer" and vals="choice text")
        for col in maps.keys():
            parsed_choices = choices.parseString(choice_strings[col]).asList()
            maps[col].update({int_: txt_ for int_, txt_ in parsed_choices})

        self.choices_map = maps

    def _make_redcap_validation_table(self):
        """Return a dataframe representing the validation columns of the ``data_dict``.

        Relevant columns: ['Text Validation Type OR Show Slider Number',
                           'Text Validation Min',
                           'Text Validation Max']

        Modifications:
            - Columns renamed to: ['type','min','max'].
            - Rows where all values are null are dropped.
            - Columns where 'type' == null are corrected as rationally as possible.
            - Values in the ['min','max'] columns are cast into correct types where possible.

        Returns:
            pandas.DataFrame
        """
        # Set up constants and stuff
        ## Missing type info
        fix = Munch()
        fix.labs = {'alb', 'crp', 'esr', 'hct', 'plt', 'wbc'}

        ## Conversion map for max/min values
        type_conversions = Munch()
        type_conversions.date_mdy = pd.Timestamp
        type_conversions.integer = np.int64
        type_conversions.number = np.float64
        type_conversions.number_1dp = np.float64
        type_conversions.date_dmy = pd.Timestamp

        # Subset and rename the target columns
        validation = self.data_dict[
            ['Text Validation Type OR Show Slider Number', 'Text Validation Min', 'Text Validation Max']].dropna(
            how='all')
        validation.columns = ['type', 'min', 'max']

        # fix the null typed rows
        ## lab values should be numbers
        validation.loc[list(fix.labs), 'type'] = 'number'

        # recast the min/max values as appropriate (IGNORING nulls for now).
        for typ, cast_func in type_conversions.items():
            idxs = validation.query(""" type == '{typ}' """.format(typ=typ)).index

            validation.loc[idxs, 'min'] = validation.loc[idxs, 'min'].apply(cast_func_ignore_nulls, f=cast_func).astype(
                'object')
            validation.loc[idxs, 'max'] = validation.loc[idxs, 'max'].apply(cast_func_ignore_nulls, f=cast_func).astype(
                'object')

        self.validation_table = validation

    def _infer_crude_dtypes(self, data_path):
        """Use data_dict to infer correct data types for each data column.

        Args:
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
        redcap_types = self.data_dict['Field Type']

        # Use redcap_types + data_col_names to associate a reasonible dtype
        col_redcap_types = Munch()

        ## distribute redcap_type to each col
        for col in data_col_names:
            try:
                col_redcap_types[col] = redcap_types[col.split('___')[0]]
            except KeyError as exc:
                # TODO: make use_category robust to different redcap sources
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
        crude_dtypes.numpy_dtypes = {col_name: rc2np[val] for col_name, val in col_redcap_types.items()}

        self.crude_dtypes = crude_dtypes

    def _load_redcap_dump(self, data_path):
        """Return loaded, recode, and validated dump table.

        Args:
            data_path (Path): Location of redcap data dump.
            self.data_dict (pandas.DataFrame): Loaded data_dict object.

        Returns:
            pandas.DataFrame
        """
        data = pd.read_csv(data_path, dtype=self.crude_dtypes.numpy_dtypes, index_col=None)

        recast_advanced_dtypes(data=data, data_dict=self.data_dict, crude_dtypes=self.crude_dtypes)

        self.data = data.set_index(['subid', 'redcap_event_name']).sort_index(level='subid')
        
        

########################## True Recoding Functions ##########################

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
    """Perform in-place re-casting of ``df[col]`` to ``pendulum.Pendulum``.
    
    Ignores nulls.
    
    Args:
        df (pandas.DataFrame): a dataframe.
        col (str): column name in ``df`` to be re-cast.
    """
    df.loc[:,col] = df[col].apply(cast_func_ignore_nulls, f=pd.Timestamp)


def cast_column_as_category(df, col):
    """Perform in-place re-casting of ``df[col]`` to ``pendulum.Pendulum``.
    
    Args:
        df (pandas.DataFrame): a dataframe.
        col (str): column name in ``df`` to be re-cast.
    """
    df[col] = df[col].astype('category')


def recast_advanced_dtypes(data, data_dict, crude_dtypes):
    """Convert dataframe columns to more useful dtypes.

    This includes using things like ``pandas.Categorical`` and ``pendulum.Pendulum`` types
    rather that defaulting to ``str`` or ``float`` primitives.

    Args:
        self.data (pandas.DataFrame): Loaded and crudely re-typed redcap data dump.
        self.data_dict (pandas.DataFrame): Loaded self.data_dict object.
        self.crude_dtypes (Munch): Output from ``_infer_crude_dtypes()``.

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
                   "category", }

    recast_as.category = [col for col, val in crude_dtypes.redcap_dtypes.items() if val in redcap_cats]

    ## Date Columns
    dt_vals = {'date_mdy', 'date_dmy'}
    recast_as.date = list(data_dict[data_dict['Text Validation Type OR Show Slider Number'].isin(dt_vals)].index.values)

    # Do the re-casts
    ## Categories
    [cast_column_as_category(df=data, col=c) for c in recast_as.category]

    ## Dates
    [cast_column_as_date(df=data, col=c) for c in recast_as.date]
