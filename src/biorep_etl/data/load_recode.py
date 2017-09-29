#!/usr/bin/env python
"""Provide logic to import, parse, and recode project data into usable structures for analysis."""

# Imports
import os
from pathlib import Path
from collections import defaultdict, namedtuple
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from munch import Munch, munchify

import engarde.decorators as ed
import engarde.checks as ck

import biorep_etl.data.parsers.pyparsers as parsers
import biorep_etl.errors as e

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"

# Constants
def tree():
    return defaultdict(tree)


# Classes
SchemaRow = namedtuple(typename="SchemaRow", field_names=["table_name", "column_name", "dtype", "categories"], verbose=False, rename=False)


class BaseData(object):
    """Organize the common loading, preparation, and storing of generic data dumps."""

    conf = None

    def __init__(self, data_path, data_dict_path):
        """Load, recode, and verify various sets of information related to a generic data dump.

        Args:
            data_path (Path): Location of csv dump.
            data_dict_path (Path): Location of data_dict csv.
        """
        self.schema_table = None

        self._make_type_conversions()
        self._load_data_dict(data_dict_path=data_dict_path)
        self._make_required_columns()
        self._infer_crude_dtypes(data_path=data_path)
        self._make_field_map()
        self._make_choice_maps()
        self._make_validation_table()
        self._load_data_dump(data_path=data_path)

        self.prep_for_sql = Munch()
        self.ready_for_sql = Munch()


    def build_tables_for_sql(self):
        """Extract and store data as tables in preparation for SQL conversion."""
        raise NotImplementedError('Override this method as appropriate in subclasses.')

    def _make_type_conversions(self):
        """Establish common map for type conversion functions."""
        raise NotImplementedError('Override this method as appropriate in subclasses.')

    def _make_required_columns(self):
        """Return list of columns that are required."""
        raise NotImplementedError('Override this method as appropriate in subclasses.')
        # self.required_columns = list(self.data_dict[self.data_dict['Required Field?'] == 'y'].index.values)

    def _load_data_dict(self, data_dict_path):
        """Load data dict into dataframe."""
        raise NotImplementedError('Override this method as appropriate in subclasses.')

        # set obj value after resetting the original order and setting the index column
        self.data_dict = pd.concat(dfs).sort_index().set_index('Variable / Field Name')


    def _make_field_map(self):
        """Return Dict to translate field names and field labels."""
        raise NotImplementedError('Override this method as appropriate in subclasses.')

        self.field_map = d

    def _make_choice_maps(self):
        """Return dict of dicts to translate ``int`` values to their meanings.

        Generally, the columns will be those whose data type is categorical.
        Examples: 'prior_protocol_number', 'gender', etc.


        Returns:
            Munch: dict of dicts -> key1 = column_names, key2 = ``int``, value = choice text.
        """
        raise NotImplementedError('Override this method as appropriate in subclasses.')

        self.choices_map = maps

    def _infer_crude_dtypes(self, data_path):
        """Use data_dict to infer correct data types for each data column.

        Args:
            data_path (Path | str): Location of data dump.

        Returns:
            Munch: ``keys in ['', '']``
        """
        raise NotImplementedError('Override this method as appropriate in subclasses.')

        self.crude_dtypes = crude_dtypes


    def _make_validation_table(self):
        """Return a dataframe representing the validation columns of the ``data_dict``.

        Relevant columns: ['Text Validation Type OR Show Slider Number',
                           'Text Validation Min',
                           'Text Validation Max']

        Modifications:
            - Columns renamed to: ['type', 'min', 'max'].
            - Rows where all values are null are dropped.
            - Columns where 'type' == null are corrected as rationally as possible.
            - Values in the ['min', 'max'] columns are cast into correct types where possible.

        Returns:
            pandas.DataFrame
        """
        raise NotImplementedError('Override this method as appropriate in subclasses.')

        self.validation_table = validation

    def _load_data_dump(self, data_path):
        """Return loaded, recode, and validated dump table.

        Args:
            data_path (Path): Location of redcap data dump.
            self.data_dict (pandas.DataFrame): Loaded data_dict object.

        Returns:
            pandas.DataFrame
        """
        raise NotImplementedError('Override this method as appropriate in subclasses.')

        self.data = self.data.sort_index()

    def _build_schema_table(self):
        """Create a table representing the dtypes for each column in each table stored in ``self.prep_for_sql``."""
        schema_table = []

        for table_name, table in self.prep_for_sql.items():
            for column_name in list(table.columns):
                dtype = table[column_name].dtype

                if dtype == 'category':
                    categories = str(list(table[column_name].cat.categories))
                else:
                    categories = str([])

                schema_table.append(SchemaRow(table_name=table_name,
                                              column_name=column_name,
                                              dtype=dtype,
                                              categories=categories)
                                    )

        raise NotImplementedError()
        self.schema_table = schema_table

    def to_csv_tables(self, directory):
        """Save DataFrames to a series of respective csv files plus a "schema" table.

        ``directory`` and parents will be created if it does not exist.

        directory (``Path``): Path to a directory where the tables will be written.
        """
        if self.schema_table is None:
            self._build_schema_table()

        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=False)

        raise NotImplementedError()


class RedCapData(BaseData):
    """Organize the common loading, preparation, and storing of RedCap data dumps."""

    conf = None

    def __init__(self, data_path, data_dict_path):
        """Load, recode, and verify various sets of information related to a RedCap dump.

        Args:
            data_path (Path): Location of redcap csv dump.
            data_dict_path (Path): Location of data_dict csv.
        """
        super().__init__(data_path, data_dict_path)

    def _make_type_conversions(self):
        """Establish common map for redcap type conversion functions."""
        type_conversions = Munch()
        type_conversions.date_mdy = pd.Timestamp
        type_conversions.integer = np.float64  # have to use float bc np.integer columns cant have NaN! `:(
        type_conversions.number = np.float64
        type_conversions.number_1dp = np.float64
        type_conversions.date_dmy = pd.Timestamp
        type_conversions.yesno = np.float64
        self.type_conversions = type_conversions

    def _make_required_columns(self):
        """Return list of columns that are required."""
        self.required_columns = list(self.data_dict[self.data_dict['Required Field?'] == 'y'].index.values)

    def _load_data_dict(self, data_dict_path):
        """Load data dict into dataframe."""
        data_dict = pd.read_csv(data_dict_path, index_col=None)

        # do not retain rows that are "descriptive" type (they are useless and cause errors)
        desc_rows = data_dict['Field Type'].isin(['descriptive'])
        data_dict = data_dict[~desc_rows]

        recode_yesno_choice_values(df=data_dict)

        # front-fill column "Section Header" using groupby to prevent filling past end of "Form Name"
        # WARNING: this is NOT perfect
        dd_groups = data_dict.groupby(['Form Name'])

        dfs = []
        for index in dd_groups.groups.values():
            df = data_dict.loc[index]
            df['Section Header'].fillna(method='ffill', inplace=True)
            dfs.append(df)

        # set obj value after resetting the original order and setting the index column
        self.data_dict = pd.concat(dfs).sort_index().set_index('Variable / Field Name')

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

        Examples:
            'prior_protocol_number', 'gender', etc.

        Returns:
            Munch: dict of dicts -> key1 = column_names, key2 = ``int``, value = choice text.
        """
        # remove any column_names that have "field type" == 'calc'
        choice_strings = self.data_dict[self.data_dict['Field Type'] != 'calc']

        # This is all items in the choice definitions column, that are not null, indexed by the data column they pertain to.
        choice_strings = choice_strings[choice_strings['Choices, Calculations, OR Slider Labels'].notnull()]['Choices, Calculations, OR Slider Labels']

        # Add first level of tree (keys=col_names and vals=Munch())
        maps = Munch({col: Munch() for col in choice_strings.index.values})

        # Create parser for choice strings
        choices = parsers.build_choices_parser()

        # add second level of tree "maps.col_name" (keys="choice_integer" and vals="choice text")
        for col in maps.keys():
            parsed_choices = choices.parseString(choice_strings[col]).asList()
            maps[col].update({int_: txt_ for int_, txt_ in parsed_choices})

        self.choices_map = maps

    def _infer_crude_dtypes(self, data_path):
        """Use data_dict to infer correct data types for each data column.

        Args:
            data_path (Path | str): Location of redcap data dump.

        Returns:
            Munch: ``keys in ['', '']``
        """
        # Define mapping from redcap type to np.dtype
        rc2np = Munch()
        rc2np.radio = np.float64
        rc2np.text = str
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

        # Use redcap_types + data_col_names to associate a reasonable dtype
        col_redcap_types = Munch()

        ## distribute redcap_type to each col
        for col in data_col_names:
            try:
                col_redcap_types[col] = redcap_types[col.split('___')[0]]
            except KeyError as exc:
                # TODO: make use_category robust to different redcap sources
                use_category = self.conf.INFER_CRUDE_DTYPES.USE_CATEGORY

                missing_key = exc.args[0]
                if missing_key in use_category:
                    col_redcap_types[col] = 'category'
                else:
                    raise exc

        # Set up return dict
        crude_dtypes = Munch()
        crude_dtypes.redcap_dtypes = col_redcap_types
        crude_dtypes.numpy_dtypes = Munch({col_name: rc2np[val] for col_name, val in col_redcap_types.items()})

        self.crude_dtypes = crude_dtypes


    def _make_validation_table(self):
        """Return a dataframe representing the validation columns of the ``data_dict``.

        Relevant columns: ['Text Validation Type OR Show Slider Number',
                           'Text Validation Min',
                           'Text Validation Max']

        Modifications:
            - Columns renamed to: ['type', 'min', 'max'].
            - Rows where all values are null are dropped.
            - Columns where 'type' == null are corrected as rationally as possible.
            - Values in the ['min', 'max'] columns are cast into correct types where possible.

        Returns:
            pandas.DataFrame
        """
        # Set up constants and stuff
        ## Missing type info
        missing_rcap_type = self.conf.MAKE_REDCAP_VALIDATION_TABLE.MISSING_RCAP_TYPE

        # Subset and rename the target columns
        validation = self.data_dict[['Text Validation Type OR Show Slider Number', 'Text Validation Min', 'Text Validation Max']].dropna(how='all')
        validation.columns = ['type', 'min', 'max']

        # fix the null typed rows
        ## lab values should be numbers for example
        for rcap_type, col_names in missing_rcap_type.items():
            validation.loc[col_names, 'type'] = rcap_type

        # recast the min/max values as appropriate (IGNORING nulls for now).
        for typ, cast_func in self.type_conversions.items():
            idxs = validation.query(""" type == '{typ}' """.format(typ=typ)).index

            validation.loc[idxs, 'min'] = validation.loc[idxs, 'min'].apply(cast_func_ignore_nulls, f=cast_func).astype('object')
            validation.loc[idxs, 'max'] = validation.loc[idxs, 'max'].apply(cast_func_ignore_nulls, f=cast_func).astype('object')

        self.validation_table = validation

    def _load_data_dump(self, data_path):
        """Return loaded, recode, and validated dump table.

        Args:
            data_path (Path): Location of redcap data dump.
            self.data_dict (pandas.DataFrame): Loaded data_dict object.

        Returns:
            pandas.DataFrame
        """
        index_cols = self.conf.LOAD_DATA_DUMP.INDEX_COLS

        data = pd.read_csv(data_path, dtype=self.crude_dtypes.numpy_dtypes, index_col=None)

        recast_advanced_dtypes(data=data, data_dict=self.data_dict, crude_dtypes=self.crude_dtypes)

        self.data = data

        if index_cols:
            self.data = self.data.set_index(index_cols)

        self.data = self.data.sort_index()


class RegistryRedCapData(RedCapData):
    """Organize the Registry-specific loading, preparation, and storing of RedCap data dumps."""

    conf = tree()
    conf['INFER_CRUDE_DTYPES']['USE_CATEGORY'] = {'redcap_event_name',
                                                  'registration_visit_complete',
                                                  'baseline_and_follow_up_complete',
                                                  'surgeries_complete',
                                                  'hospitalizations_complete'}

    conf['MAKE_REDCAP_VALIDATION_TABLE']['MISSING_RCAP_TYPE']['number'] = ['alb', 'crp', 'esr', 'hct', 'plt', 'wbc']

    conf['LOAD_DATA_DUMP']['INDEX_COLS'] = ['subid', 'redcap_event_name']

    conf = munchify(conf)

    def __init__(self, data_path, data_dict_path):
        """Load, recode, and verify various sets of information related to a HARVARD_REGISTRY RedCap dump.

        Args:
            data_path (Path): Location of redcap csv dump.
            data_dict_path (Path): Location of data_dict csv.
        """
        super().__init__(data_path, data_dict_path)

    def build_tables_for_sql(self):
        """Extract and store data as tables in preparation for SQL conversion."""
        self.prep_for_sql.update(self._build_subject_table())
        self.prep_for_sql.update(self._build_jewish_ancetry_table())
        self.prep_for_sql.update(self._build_baseline_and_follow_up_table())
        self.prep_for_sql.update(self._build_family_and_birth_history_table())
        self.prep_for_sql.update(self._build_smoking_and_alcohol_use_table())
        self.prep_for_sql.update(self._build_growth_and_development_table())
        self.prep_for_sql.update(self._build_ibd_disease_course_table())
        self.prep_for_sql.update(self._build_cd_characteristics_table())
        self.prep_for_sql.update(self._build_pcdai_table())
        self.prep_for_sql.update(self._build_hbai_table())
        self.prep_for_sql.update(self._build_uc_ic_characteristics_table())
        self.prep_for_sql.update(self._build_pucai_table())
        self.prep_for_sql.update(self._build_colon_cancer_dysplasia_history_table())
        self.prep_for_sql.update(self._build_ibd_meds_history_table())
        self.prep_for_sql.update(self._build_discontinued_ibd_meds_table())
        self.prep_for_sql.update(self._build_adverse_event_intolerance_meds_table())
        self.prep_for_sql.update(self._build_labs_table())
        self.prep_for_sql.update(self._build_environmental_and_suppl_questions_table())
        self.prep_for_sql.update(self._build_veoibd_questions_table())
        self.prep_for_sql.update(self._build_surgeries_table())
        self.prep_for_sql.update(self._build_hospitalizations_table())

    #
    # def _build_XXX_table(self):
    #     """Gather and return dataframe representing the xxxxxxx tables.
    #
    #     Returns:
    #         dict-like: tables ["??", "??"]
    #     """
    #     parent_table_name = 'xxxxxxxxxxx'
    #
    #     checkboxes = {}
    #
    #     cols = []
    #
    #     return build_generic_table(obj=self,
    #                                parent_table_name=parent_table_name,
    #                                checkboxes=checkboxes,
    #                                cols=cols)



    def _build_baseline_and_follow_up_table(self):
        """Gather and return dataframe representing the baseline_and_follow_up tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'baseline_and_follow_up'

        checkboxes = {'historyfrom': None}

        cols = ['visitcategory',
                'paperfill',
                'entertoredcap',
                'review',
                'dateentered',
                'ibddiag1',
                'ageatdiag',
                'visittype',
                'onsetdt',
                'ageatonset',
                'ibdtype',
                'gisurg',
                'ibdhosp',
                'ibdmed',
                'enviquest',
                'ibddiagchange',
                'prioribddiag',
                'newidbdiag',
                'ancestrynew',
                'famhisnew',
                'gisurgnew']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)




    def _build_jewish_ancetry_table(self):
        """Gather and return dataframe representing the jewish_ancetry tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'jewish_ancetry'

        checkboxes = {'jewanctype': None}

        cols = ['jeworigin', 'jeworiginnew']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)


    def _build_colon_cancer_dysplasia_history_table(self):
        """Gather and return dataframe representing the colon_cancer_dysplasia_history tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'colon_cancer_dysplasia_history'

        checkboxes = {}

        cols = ['hxcolcanc',
                'hxpolyps',
                'hxcoldysp',
                'coldysptype',
                'coldyspgrade']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_pucai_table(self):
        """Gather and return dataframe representing the pucai tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'pucai'

        checkboxes = {'pucaiobt': None}

        cols = ['pucaidoa',
                'pucaiabpain',
                'pucairectal',
                'pucaiconsist',
                'pucaifreq',
                'pucaiawaken',
                'pucaiactivity',
                'pucaiscore',
                'pucaiseverity']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)


    def _build_uc_ic_characteristics_table(self):
        """Gather and return dataframe representing the uc_ic_characteristics tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'uc_ic_characteristics'

        checkboxes = {'ucstudies': 'ucstudiesoth',
                      'procevi': None,
                      'lsideevi': None,
                      'pancoevi': None}

        cols = ['ucvisit',
                'ucvisit2',
                'proctdis',
                'lsidedis',
                'pancodis',
                'parisuc',
                'parissev']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_hbai_table(self):
        """Gather and return dataframe representing the hbai tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'hbai'

        checkboxes = {'hbaiobt': None,
                      'hbaicomp': None}

        cols = ['hbaidoa',
                'hbaiwell',
                'hbaiabpain',
                'hbaistoolnum',
                'hbaiabmass',
                'hbaicomp1',
                'hbaiscore']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_ibd_disease_course_table(self):
        """Gather and return dataframe representing the ibd_disease_course tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        # TODO: decide whether [ibdcoex_type,ibdcoex_other] ought to be treated like a checkbox group
        parent_table_name = 'ibd_disease_course'

        checkboxes = {'ibdliverl': None,
                      'ibdmanl': 'otherextraintest',
                      'ibdcondl': 'otherauto'}

        cols = ['ibdliver',
                'ibdman',
                'ibdcond',
                'ibdcoex',
                'ibdcoex_type',
                'ibdcoex_other']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_growth_and_development_table(self):
        """Gather and return dataframe representing the growth_and_development tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'growth_and_development'

        checkboxes = {}

        cols = ['wtdiag',
                'wtdiagdt',
                'htdiag',
                'htdiagdt',
                'prevwtdiag',
                'prevwtdiagdt',
                'prevhtdiag',
                'prevhtdiagdt']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_smoking_and_alcohol_use_table(self):
        """Gather and return dataframe representing the smoking_and_alcohol_use tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'smoking_and_alcohol_use'

        checkboxes = {}

        cols = ['tobacco',
                'tobaccopacks',
                'tobaccowhen',
                'tobaccoyes',
                'alcohol',
                'alcoholdrinks',
                'alcoholwhen',
                'alcoholyes']

        m =  build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

        # Convert appropriate columns to numbers
        mkfloat = ["tobaccowhen", "alcoholwhen"]
        for col in mkfloat:
            m[parent_table_name][col] = m[parent_table_name][col].astype(np.float64)

        return m

    def _build_family_and_birth_history_table(self):
        """Gather and return dataframe representing the family_and_birth_history tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'family_and_birth_history'

        checkboxes = {'famcd': None,
                      'famuc': None,
                      'famun': None,
                      'famautotype': 'famautooth',
                      'famimmtype': 'famimmoth'}

        cols = ['anyfamauto',
                'twincdtype',
                'twinuctype',
                'tineuntype',
                'famauto',
                'famimm',
                'famcolcanc',
                'fambreast',
                'breastdur',
                'breastdurex',
                'caesarean']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_hospitalizations_table(self):
        """Gather and return dataframe representing the hospitalizations tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'hospitalizations'

        checkboxes = {}

        cols = ['hosp_a',
                'hosptype_a',
                'hospibdtype_a',
                'hosp_b',
                'hosptype_b',
                'hospibdtype_b',
                'hosp_c',
                'hosptype_c',
                'hospibdtype_c',
                'hosp_d',
                'hosptype_d',
                'hospibdtype_d',
                'hosp_e',
                'hosptype_e',
                'hospibdtype_e',
                'hosp_f',
                'hosptype_f',
                'hospibdtype_f',
                'hosp_g',
                'hosptype_g',
                'hospibdtype_g',
                'hosp_h',
                'hosptype_h',
                'hospibdtype_h',
                'hosp_i',
                'hosptype_i',
                'hospibdtype_i',
                'hosp_j',
                'hosptype_j',
                'hospibdtype_j',
                'hosp_k',
                'hosptype_k',
                'hospibdtype_k',
                'hosp_l',
                'hosptype_l',
                'hospibdtype_l',
                'hosp_m',
                'hosptype_m',
                'hospibdtype_m']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_surgeries_table(self):
        """Gather and return dataframe representing the surgeries tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'surgeries'

        checkboxes = {'surgerytype_a': None,
                      'surgerytype_b': None,
                      'surgerytype_c': None,
                      'surgerytype_d': None,
                      'surgerytype_e': None,
                      'surgerytype_f': None,
                      'surgerytype_g': None,
                      'surgerytype_h': None,
                      'surgerytype_i': None,
                      'surgerytype_j': None,
                      'surgerytype_k': None,
                      'surgerytype_l': None}

        cols = ['surg_a',
                'surg_b',
                'surg_c',
                'surg_d',
                'surg_e',
                'surg_f',
                'surg_g',
                'surg_h',
                'surg_i',
                'surg_j',
                'surg_k',
                'surg_l']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_veoibd_questions_table(self):
        """Gather and return dataframe representing the veoibd_questions tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'veoibd_questions'

        checkboxes = {'crite': None,
                      'oral_perianal': None}

        cols = ['country',
                'dateofarrival',
                'follic',
                'milkal',
                'consanguinity',
                'nsaid',
                'nsaid_doses',
                'gastroinfect']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_environmental_and_suppl_questions_table(self):
        """Gather and return dataframe representing the environmental_and_suppl_questions tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'environmental_and_suppl_questions'

        checkboxes = {}

        cols = ['smokeprior',
                'smokeprior2',
                'smokepreg',
                'antibioticprior',
                'antibioticpriormo',
                'steroidprior',
                'steroidpriormo']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_adverse_event_intolerance_meds_table(self):
        """Gather and return dataframe representing the adverse_event_intolerance_meds tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'adverse_event_intolerance_meds'

        checkboxes = {}

        cols = ['meddraae1',
                'meddraae2',
                'meddraae3',
                'meddraae4',
                'meddraae5',
                'meddraae6',
                'meddraae7',
                'meddraae8',
                'meddraae9',
                'meddraae10',
                'meddraae11',
                'meddraae12',
                'meddraae13',
                'meddraae14',
                'meddraae15',
                'meddraae166',
                'meddraae16',
                'meddraae30',
                'meddraae17',
                'meddraae18',
                'meddraae19',
                'meddraae20',
                'meddraae21',
                'meddraae22',
                'meddraae23',
                'meddraae24',
                'meddraae25',
                'meddraae26',
                'meddraae27',
                'meddraae28',
                'meddraae29',
                'othermedraae']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_discontinued_ibd_meds_table(self):
        """Gather and return dataframe representing the discontinued_ibd_meds tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'discontinued_ibd_meds'

        checkboxes = {'medinfosc2': None}

        cols = ['mesalamine2',
                'sulfasa2',
                'balsalazide2',
                'olsalazine2',
                'mercap62',
                'azathioprine2',
                'methotrex2',
                'fk5062',
                'cyclospa2',
                'thalidomide2',
                'lenalidomide2',
                'infliximab2',
                'adalimumab2',
                'certolizumab2',
                'golimumab2',
                'vedolizumab2',
                'natalizumab2',
                'tocilizumab2',
                'ustekinumab2',
                'oralenteric2',
                'rectalstrd2',
                'oralsterd2',
                'intrasterd2',
                'metronida2',
                'ciproflox2',
                'trimethop2',
                'amoxicillin2',
                'vancomycin2',
                'ursodiol2',
                'cholestyr2',
                'rifampin2']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_ibd_meds_history_table(self):
        """Gather and return dataframe representing the ibd_meds_history tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        # TODO: test that "No, never been on" never ocurs AFTER previous "Yes, current or past use" response
        # TODO: examine whether to subtable out examples of the drug classes
        parent_table_name = 'ibd_meds_history'

        checkboxes = {'medinfosc': None}

        cols = ['aminosali',
                'mesalamine',
                'sulfasa',
                'balsalazide',
                'olsalazine',
                'chemomed',
                'mercap6',
                'azathioprine',
                'methotrex',
                'fk506',
                'cyclospa',
                'thalidomide',
                'lenalidomide',
                'antitnfmed',
                'infliximab',
                'adalimumab',
                'certolizumab',
                'golimumab',
                'othbiomed',
                'natalizumab',
                'tocilizumab',
                'vedolizumab',
                'ustekinumab',
                'glucotds',
                'oralenteric',
                'rectalstrd',
                'oralsterd',
                'intrasterd',
                'steroidcourse',
                'antibiotics',
                'metronida',
                'ciproflox',
                'trimethop',
                'amoxicillin',
                'vancomycin',
                'pscmeds',
                'ursodiol',
                'cholestyr',
                'rifampin',
                'enteral',
                'supplements',
                'omega3',
                'probiotic',
                'fecaltrans']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)




    def _build_cd_characteristics_table(self):
        """Gather and return dataframe representing the cd_characteristics tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'cd_characteristics'

        checkboxes = {'chohnstudies': 'chohnstudiesoth',
                      'esoevi': None,
                      'stomevi': None,
                      'duodevi': None,
                      'jejuevi': None,
                      'ileumevi': None,
                      'cecumevi': None,
                      'transevi': None,
                      'descevi': None,
                      'sigmevi': None,
                      'rectumevi': None,
                      'parisloc': None}

        cols = ['crohnvisit',
                'crohnvisit2',
                'esoact',
                'stomact',
                'duodact',
                'jejuact',
                'ileumact',
                'cecumact',
                'transact',
                'descact',
                'sigmact',
                'rectumact',
                'orophary',
                'perianal',
                'skin',
                'fissure',
                'fistula',
                'perianal_stricture',
                'mucosal2',
                'stenosing',
                'fistulizing',
                'enteric',
                'toskin',
                'scrotal',
                'bladder',
                'abcessess',
                'parisclass',
                'parisbeh',
                'parisgrow']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_pcdai_table(self):
        """Gather and return dataframe representing the pcdai tables.

        Returns:
            dict-like: tables ["pcdaiobt", "pcdai"]
        """
        parent_table_name = "pcdai"
        checkboxes = None
        cols = ["pcdaidoa", "pcdaiabpain",
                "pcdaistools", "pcdaifunction", "pcdaiwt",
                "pcdaitender", "pcdairectal", "pcdaimanifest",
                "pcdaiscore", "pcdaiseverity"]

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_subject_table(self):
        """Gather and return dataframe representing the subject table."""
        # TODO: add jewish ancestry columns
        parent_table_name = "subject"
        checkboxes = None
        cols = ["visitdate", "consent", "veoibd", "dob", "gender", "race", "ethnicity", "ibddiag", "contact"]

        m = build_generic_table(obj=self,
                                parent_table_name=parent_table_name,
                                checkboxes=checkboxes,
                                cols=cols)

        m.subject = m.subject.drop('redcap_event_name', axis=1)

        return m

    def _build_labs_table(self):
        """Gather and return dataframe representing the labs table."""
        # TODO: decide whether or not to keep rows with NaN for all lab results
        parent_table_name = "labs"
        checkboxes = None

        cols = ["visitcategory", "lab_reference", "date_labs",
                "alb", "alk", "ast", "bun",
                "creatine", "crp", "esr",
                "ggt", "hct", "hemoglobin",
                "plt", "wbc",]

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)


class BiorepoRedCapData(RedCapData):
    """Organize the Registry-specific loading, preparation, and storing of RedCap data dumps."""

    conf = tree()

    # TODO: configure BiorepoRedCapData CONF variable
    conf['INFER_CRUDE_DTYPES']['USE_CATEGORY'] = {"ibd_biorepository_sample_database_complete"}

    conf['MAKE_REDCAP_VALIDATION_TABLE']['MISSING_RCAP_TYPE']['number'] = []  # ['prior_protocol_number', 'ibdyesno', 'ibdtype',
                                                                              # 'ibd_immunodeficiencies', 'controldx', 'control_immunodeficiency',
                                                                              # 'familymembertype', 'familymembergidx', 'gender', 'non_bch_pt',
                                                                              # 'nsaid_doses', 'sampletype', 'longitudinal', 'timetominus20',
                                                                              # 'timetominus80', 'hbi_wellbeing', 'hbi_abdpain',
                                                                              # 'hbi_abdominalmass', 'hbi_complications', 'hbi_abscess',
                                                                              # 'hbi_fissure', 'hbi_aphthous', 'hbi_arthralgia', 'hbi_nodosum',
                                                                              # 'hbi_fistula', 'hbi_gangrenosum', 'hbi_uveitis',
                                                                              # 'sccai_stoolsperday', 'sccai_stoolspernight', 'sccai_urgency',
                                                                              # 'sccai_blood', 'sccai_wellbeing', 'sccai_arthralgia',
                                                                              # 'sccai_nodosum', 'sccai_gangrenosum', 'sccai_uveitis',
                                                                              # 'pucai_abdpain', 'pucai_bleeding', 'pucai_consistency',
                                                                              # 'pucai_frequency', 'pucai_nocturnal', 'pucai_limitation']
                                                                              #
    conf['LOAD_DATA_DUMP']['INDEX_COLS'] = 'biorepidnumber'

    conf = munchify(conf)

    def __init__(self, data_path, data_dict_path):
        """Load, recode, and verify various sets of information related to a BIOREPOSITORY RedCap dump.

        Args:
            data_path (Path): Location of redcap csv dump.
            data_dict_path (Path): Location of data_dict csv.
        """
        super().__init__(data_path, data_dict_path)

        # drop rows whose index == NaN
        not_nans = self.data.reset_index()['biorepidnumber'].notnull()
        self.data = self.data.reset_index()[not_nans].set_index('biorepidnumber')



    def build_tables_for_sql(self):
        """Extract and store data as tables in preparation for SQL conversion."""
        # self.prep_for_sql.update(self._build_family_table())
        self.prep_for_sql.update(self._build_subject_table())
        self.prep_for_sql.update(self._build_sample_table())
        self.prep_for_sql.update(self._build_visit_table())

    # def _build_XXX_table(self):
    #     """Gather and return dataframe representing the xxxxxxx tables.
    #
    #     Returns:
    #         dict-like: tables ["??", "??"]
    #     """
    #     parent_table_name = 'xxxxxxxxxxx'
    #
    #     checkboxes = {}
    #
    #     cols = []
    #
    #     return build_generic_table(obj=self,
    #                                parent_table_name=parent_table_name,
    #                                checkboxes=checkboxes,
    #                                cols=cols)

    # def _build_family_table(self):
    #     """Gather and return dataframe representing the family tables.
    #
    #     Returns:
    #         dict-like: tables ["??", "??"]
    #     """
    #     parent_table_name = 'family'
    #
    #     checkboxes = {}
    #
    #     cols = []
    #
    #     return build_generic_table(obj=self,
    #                                parent_table_name=parent_table_name,
    #                                checkboxes=checkboxes,
    #                                cols=cols)

    def _build_subject_table(self):
        """Gather and return dataframe representing the subject tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'subject'

        # TODO: go add the correct "other" cols
        checkboxes = {'familymember_ibd': None,
                      'race': None}

        cols = ['registryconsent',
                'biorepconsent',
                'ibdyesno',
                'ibdtype',
                'otheribddx',
                'family_history',
                'other_relative',
                'ibd_immunodeficiencies',
                'other_immunodeficiency',
                'controldx',
                'other_diagnosis',
                'control_immunodeficiency',
                'otherimmuno_control',
                'psc',
                'familymembertype',
                'familymembergidx',
                'family_otherdiagnosis',
                'gender',
                'dob',
                'hispanic_or_latino',
                'neopics_sample',
                'chb_id',
                'neopics',
                'bch_pt',
                'referring_hospital',
                'referring_physician',
                'symptom_onset',
                'date_of_diagnosis',
                'consanguinity']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_sample_table(self):
        """Gather and return dataframe representing the sample tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'sample'

        # TODO: go add the correct "other" cols
        checkboxes = {'blood_samples': None,
                      'stoolmedia': None,
                      'bxlocation': None,
                      'surgicallocations': None,
                      'where_stored': None}


        cols = ['record_id',
                'samplenumber',
                'label_on_sample',
                'completed_by',
                'date_crf_completed',
                'sample_date',
                'prior_protocol_number',
                'sampletype',
                'other_sampletype',
                'longitudinal',
                'timetominus20',
                'timetominus80',
                'stool_dna',
                'other_shipped',
                'labmember']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)

    def _build_visit_table(self):
        """Gather and return dataframe representing the visit tables.

        Returns:
            dict-like: tables ["??", "??"]
        """
        parent_table_name = 'visit'

        # TODO: go add the correct "other" cols
        checkboxes = {'asa_type': None,
                      'antibiotic_type': None,
                      'probiotic_type': None,
                      'glucocorticoid_type': None,
                      'immunomodulator_type': None,
                      'antitnf_type': None,
                      'calcineurin_type': None,
                      'otherbiologic_type': None,
                      'othermed_type': None}

        cols = ['samplenumber',
                'sample_date',
                'nsaid',
                'nsaid_doses',
                'hbi_wellbeing',
                'hbi_abdpain',
                'hbi_liquidstools',
                'hbi_abdominalmass',
                'hbi_complications',
                'hbi_abscess',
                'hbi_fissure',
                'hbi_aphthous',
                'hbi_arthralgia',
                'hbi_nodosum',
                'hbi_fistula',
                'hbi_gangrenosum',
                'hbi_uveitis',
                'hbi_total',
                'sccai_stoolsperday',
                'sccai_stoolspernight',
                'sccai_urgency',
                'sccai_blood',
                'sccai_wellbeing',
                'sccai_eim',
                'sccai_arthralgia',
                'sccai_nodosum',
                'sccai_gangrenosum',
                'sccai_uveitis',
                'sccai_total',
                'pucai_abdpain',
                'pucai_bleeding',
                'pucai_consistency',
                'pucai_frequency',
                'pucai_nocturnal',
                'pucai_limitation',
                'pucai_total',
                'anymeds',
                'current_5asa',
                'current_antibiotic',
                'other_antibiotic',
                'current_probiotic',
                'other_probiotic',
                'current_glucocorticoid',
                'current_immunomodulator',
                'current_antitnf',
                'current_calcineurin',
                'current_otherbiologics',
                'fecaltransplant',
                'current_othermeds',
                'othermed',
                'date_labs',
                'hct',
                'wbc',
                'plt',
                'crp',
                'esr',
                'alb',
                'notes']

        return build_generic_table(obj=self,
                                   parent_table_name=parent_table_name,
                                   checkboxes=checkboxes,
                                   cols=cols)


########################## General Helper Functions ##########################
def build_generic_table(obj, parent_table_name, cols, checkboxes=None):
    """Gather and return dataframe representing a generic table set based on provided variables.

    Args:
        obj (RedCapData or subclasses): arg info.
        parent_table_name (str): Name to be used for the table group.
        cols (list): List of column_names after removing those used in ``checkboxes``.
        checkboxes (dict): KEY = checkbox_column_name,
                           VALUE = other_column_name respective to KEY; else: ``None``

    Returns:
        dict-like: tables ["??", "??"]
    """
    # TODO: automatically cast columns
    # TODO: test and log when columns when they fail the validation type/min/max
    m = Munch()

    if checkboxes is not None:
        for checkbox_base, other_col in checkboxes.items():
            table_name = '{parent_table_name}_{checkbox_base}'.format(parent_table_name=parent_table_name,
                                                                      checkbox_base=checkbox_base)

            m[table_name] = make_checkbox_subtable(df=obj.data,
                                                   col_base_name=checkbox_base,
                                                   other_col=other_col,
                                                   foreign_key_cols=None,
                                                   choice_map=obj.choices_map)

    m[parent_table_name] = obj.data[cols].dropna(axis=0, how='all').copy()

    for col in cols:
        try:
            verbosify(df=m[parent_table_name], col=col, choice_map=obj.choices_map)
        except KeyError:
            pass

    for name, table in m.items():
        m[name] = table.sort_index().reset_index()

        # auto-cast each column before returning
        # BUT: do not try to REcast the subtable data columns (we already did them)
        if checkboxes is not None:
            cols_to_cast = [c for c in m[name].columns.values if c not in checkboxes.keys()]
        else:
            cols_to_cast = [c for c in m[name].columns.values]

        for col in cols_to_cast:
            try:
                cast_column_by_text_validation_type(df=m[name],
                                                    col=col,
                                                    dd=obj.data_dict,
                                                    conversion_map=obj.type_conversions)
            except KeyError:
                pass

    for name in m.keys():
        m[name] = m[name].drop_duplicates()
        
    return m


########################## True Recoding Functions ##########################
def cast_column_by_text_validation_type(df, col, dd, conversion_map):
    """Perform in-place re-casting of ``df[col]`` by infering the type from the data_dict['Text Validation Type OR Show Slider Number'].

    Args:
        df (pandas.DataFrame): a dataframe.
        col (str): column name in ``df`` to be re-cast.
        dd (dict-like): A redcap data_dict.
        conversion_map (dict-like): Maps value for the ``col`` in dd['Text Validation Type OR Show Slider Number'] to casting function.
    """
    vtype = dd.loc[col, 'Text Validation Type OR Show Slider Number']
    cast_func = conversion_map[vtype]
    df.loc[:, col] = df[col].astype(cast_func)


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
    """Perform in-place re-casting of ``df[col]`` to ``pd.Timestamp``.

    Ignores nulls.

    Args:
        df (pandas.DataFrame): a dataframe.
        col (str): column name in ``df`` to be re-cast.
    """
    def cast_time(x):
        try:
            return pd.Timestamp(x)
        except ValueError as exc:
            if "could not convert string to Timestamp" in exc.args[0]:
                msg = "{time} is not valid, setting to `NaT`.".format(time=x)
                logger.warning(msg)
                return pd.NaT



    df.loc[:,col] = df[col].apply(cast_time)


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
    recast_as.date = list(data_dict[data_dict['Text Validation Type OR Show Slider Number'].astype(str).str.startswith('date')].index.values)

    # Do the re-casts
    ## Categories
    [cast_column_as_category(df=data, col=c) for c in recast_as.category]

    ## Dates
    [cast_column_as_date(df=data, col=c) for c in recast_as.date]


def verbosify(df, col, choice_map=None):
    """Replace the cryptic values (usually integers) of a dataframe column with the verbose values.

    Args:
        df (pandas.dataframe): a dataframe.
        col (str): String used as column index in ``df``.
        choice_map (dict-like): multilevel dict-like tree with level 1 being column names and level 2 being terse to verbose maps.

    Returns:
        None: modifies ``df``.
    """
    # DONE: eliminate the need for special treatment of `yesno` columns

    df.loc[:,col] = df.loc[:,col].apply(lambda i: choice_map[col][str(int(i))])

def recode_yesno_choice_values(df):
    """Change unset "Choices, Calculations, OR Slider Labels" with ``yesno`` type to useful choice strings."""
    # Add "Choices, Calculations, OR Slider Labels" values for ``yesno`` types ONLY IF the value is NaN
    def recode(row):
        if ((row['Field Type'] == 'yesno')
            and
            (str(row['Choices, Calculations, OR Slider Labels']).upper() == 'NAN')):
            return '0, No | 1, Yes'
        else:
            return row['Choices, Calculations, OR Slider Labels']

    new_choices = df.apply(recode, axis=1)
    df['Choices, Calculations, OR Slider Labels'] = new_choices

def make_checkbox_subtable(df, col_base_name=None, other_col=None, foreign_key_cols=None, choice_map=None):
    """Return a new pandas.DataFrame obj solving the second normal form problem posed by RedCap checkbox data.

    Args:
        df (pandas.DataFrame): Parent table.
        foreign_key_cols (list | str): Column name(s) of parent table to use as the foreign key of subtable: if ``None``, use index.
        col_base_name (str): The text preceeding the ``___`` separator in checkbox group column names.
        other_col (str): If any: the text name of the "if other list here" column linked to the checkbox group.

    Returns:
        pandas.DataFrame: subtable of ``df``
    """
    if isinstance(foreign_key_cols, str):
        foreign_key_cols = [foreign_key_cols]


    main_cols = df[[col for col in df.columns.values if col.split('___')[0] == col_base_name]]

    # Excluding `other_col`, create new subtable by melting, droping, and mapping "stuff"
    # # if isinstance(main_cols.index, pd.MultiIndex):
    # #     index_names = main_cols.index.names
    # # else:
    # #     index_names = main_cols.index.names

    index_names = main_cols.index.names

    main_cols_ = pd.melt(frame=main_cols.reset_index(),
                         id_vars=index_names,
                         value_vars=None,
                         var_name=col_base_name,
                         value_name='checked_status',
                         col_level=None).set_index(index_names).sort_index()

    # collect only the checked boxes into the new subtable and map box id to values
    checked_boxes = main_cols_.checked_status == 1
    subtable = main_cols_[checked_boxes][col_base_name].apply(lambda x: x.split('___')[1]).str.replace('_', '-')
    subtable = subtable.apply(lambda x: choice_map[col_base_name][x])

    # deduplicate values in `other_col` as best as we can
    # TODO: figure out how to handle when the "other_col" contains multiple values
    # TODO: handle deduping better
    # for now enforce case is as good as I am going to try
    if other_col is not None:
        other_col_series = df[other_col].dropna().str.lower()
        other_col_series.name = col_base_name
        subtable = pd.concat([subtable,other_col_series], axis=0)

    # Cast as category and return
    return subtable.astype('category')
