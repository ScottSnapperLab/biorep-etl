#!/usr/bin/env python
"""Provide functions useable by engarde to ensure the validity of a loaded redcap dump table."""

# Imports
import os
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from munch import Munch

import engarde.decorators as ed

from biorep_etl.data import load_recode as loading
import biorep_etl.errors as e

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"

# Constants

# Helper Functions

# # Used to automate the func slug generation
# def write_func_slug(col_names, path='func_slug.py'):
#     tmpl = '''def {col_name}(df, dd):\n    """Return True if the {col_name} column tests pass."""\n    raise NotImplementedError\n\n'''
#     with open(path, 'w') as out:
#         for col_name in col_names:
#             out.write(tmpl.format(col_name=col_name))

def parse_choices(string):
    """Return a dict of numer to value mappings."""
    items = string.split('|')
    return {int(num):choice.strip() for num, choice in [i.strip().split(',') for i in items]}

def parse_validation_type(string):
    """Return a """


def is_number_Ndp(df, col, decimals=1):
    """Return series of True/False corresponding to which rows passed/failed the tests."""
    split = df[col].str.split('.', expand=True).rename(columns={0:'left',1:'right'})
    
    t = {}
    
    # test there is only one '.'
    t['only_one_dot'] = split.fillna('0').apply(lambda row: len(row) == 2, axis=1)
    
    # test the strings are int-able
    t['left_ints'] =  split.left.fillna('0').apply(lambda i: isinstance(np.int64(i),np.int64))
    t['right_ints'] =  split.right.fillna('0').apply(lambda i: isinstance(np.int64(i),np.int64))
    
    # test the right side is 4 digits long
    t['right_4dp'] =  split.right.fillna('0' * decimals).apply(lambda i: len(i) == decimals)
    
    t = pd.DataFrame(t)
    return t.apply(lambda row: row.all(), axis=1)

def is_in_range_obj_as_float(df, col, lo, hi):
    pass
    
def req_cols(dd):
    """Return list of columns that are required."""
    return list(dd[dd['Required Field?'] == 'y'].index.values)

# Column Test Functions

tests = Munch()

def record_id(df, dd):
    """Return True if the record_id column tests pass."""
    raise NotImplementedError


@ed.verify_all(is_number_Ndp, col='samplenumber', decimals=4)
def samplenumber(df, dd):
    """Return True if the samplenumber column tests pass."""
    info = dd.loc['samplenumber']

    pass
tests.samplenumber = samplenumber

def label_on_sample(df, dd):
    """Return True if the label_on_sample column tests pass."""
    raise NotImplementedError

def mrn(df, dd):
    """Return True if the mrn column tests pass."""
    raise NotImplementedError


@ed.verify_all(is_number_Ndp, col='samplenumber', decimals=1)
def biorepidnumber(df, dd):
    """Return True if the biorepidnumber column tests pass."""
    raise NotImplementedError

def completed_by(df, dd):
    """Return True if the completed_by column tests pass."""
    raise NotImplementedError

def date_crf_completed(df, dd):
    """Return True if the date_crf_completed column tests pass."""
    raise NotImplementedError

def sample_date(df, dd):
    """Return True if the sample_date column tests pass."""
    raise NotImplementedError

def registryconsent(df, dd):
    """Return True if the registryconsent column tests pass."""
    raise NotImplementedError

def biorepconsent(df, dd):
    """Return True if the biorepconsent column tests pass."""
    raise NotImplementedError

def prior_protocol_number(df, dd):
    """Return True if the prior_protocol_number column tests pass."""
    raise NotImplementedError

def ibdyesno(df, dd):
    """Return True if the ibdyesno column tests pass."""
    raise NotImplementedError

def ibdtype(df, dd):
    """Return True if the ibdtype column tests pass."""
    raise NotImplementedError

def otheribddx(df, dd):
    """Return True if the otheribddx column tests pass."""
    raise NotImplementedError

def family_history(df, dd):
    """Return True if the family_history column tests pass."""
    raise NotImplementedError

def familymember_ibd(df, dd):
    """Return True if the familymember_ibd column tests pass."""
    raise NotImplementedError

def other_relative(df, dd):
    """Return True if the other_relative column tests pass."""
    raise NotImplementedError

def ibd_immunodeficiencies(df, dd):
    """Return True if the ibd_immunodeficiencies column tests pass."""
    raise NotImplementedError

def other_immunodeficiency(df, dd):
    """Return True if the other_immunodeficiency column tests pass."""
    raise NotImplementedError

def controldx(df, dd):
    """Return True if the controldx column tests pass."""
    raise NotImplementedError

def other_diagnosis(df, dd):
    """Return True if the other_diagnosis column tests pass."""
    raise NotImplementedError

def control_immunodeficiency(df, dd):
    """Return True if the control_immunodeficiency column tests pass."""
    raise NotImplementedError

def otherimmuno_control(df, dd):
    """Return True if the otherimmuno_control column tests pass."""
    raise NotImplementedError

def psc(df, dd):
    """Return True if the psc column tests pass."""
    raise NotImplementedError

def familymembertype(df, dd):
    """Return True if the familymembertype column tests pass."""
    raise NotImplementedError

def familymembergidx(df, dd):
    """Return True if the familymembergidx column tests pass."""
    raise NotImplementedError

def family_otherdiagnosis(df, dd):
    """Return True if the family_otherdiagnosis column tests pass."""
    raise NotImplementedError

def gender(df, dd):
    """Return True if the gender column tests pass."""
    raise NotImplementedError

def dob(df, dd):
    """Return True if the dob column tests pass."""
    raise NotImplementedError

def age(df, dd):
    """Return True if the age column tests pass."""
    raise NotImplementedError

def neopics_sample(df, dd):
    """Return True if the neopics_sample column tests pass."""
    raise NotImplementedError

def chb_id(df, dd):
    """Return True if the chb_id column tests pass."""
    raise NotImplementedError

def neopics(df, dd):
    """Return True if the neopics column tests pass."""
    raise NotImplementedError

def symptom_onset(df, dd):
    """Return True if the symptom_onset column tests pass."""
    raise NotImplementedError

def date_of_diagnosis(df, dd):
    """Return True if the date_of_diagnosis column tests pass."""
    raise NotImplementedError

def consanguinity(df, dd):
    """Return True if the consanguinity column tests pass."""
    raise NotImplementedError

def oral_perianal(df, dd):
    """Return True if the oral_perianal column tests pass."""
    raise NotImplementedError

def nsaid(df, dd):
    """Return True if the nsaid column tests pass."""
    raise NotImplementedError

def nsaid_doses(df, dd):
    """Return True if the nsaid_doses column tests pass."""
    raise NotImplementedError

def sampletype(df, dd):
    """Return True if the sampletype column tests pass."""
    raise NotImplementedError

def other_sampletype(df, dd):
    """Return True if the other_sampletype column tests pass."""
    raise NotImplementedError

def longitudinal(df, dd):
    """Return True if the longitudinal column tests pass."""
    raise NotImplementedError

def blood_samples(df, dd):
    """Return True if the blood_samples column tests pass."""
    raise NotImplementedError

def timetominus20(df, dd):
    """Return True if the timetominus20 column tests pass."""
    raise NotImplementedError

def timetominus80(df, dd):
    """Return True if the timetominus80 column tests pass."""
    raise NotImplementedError

def stoolmedia(df, dd):
    """Return True if the stoolmedia column tests pass."""
    raise NotImplementedError

def stool_dna(df, dd):
    """Return True if the stool_dna column tests pass."""
    raise NotImplementedError

def bxlocation(df, dd):
    """Return True if the bxlocation column tests pass."""
    raise NotImplementedError

def surgicallocations(df, dd):
    """Return True if the surgicallocations column tests pass."""
    raise NotImplementedError

def where_stored(df, dd):
    """Return True if the where_stored column tests pass."""
    raise NotImplementedError

def other_shipped(df, dd):
    """Return True if the other_shipped column tests pass."""
    raise NotImplementedError

def labmember(df, dd):
    """Return True if the labmember column tests pass."""
    raise NotImplementedError

def hbi_wellbeing(df, dd):
    """Return True if the hbi_wellbeing column tests pass."""
    raise NotImplementedError

def hbi_abdpain(df, dd):
    """Return True if the hbi_abdpain column tests pass."""
    raise NotImplementedError

def hbi_liquidstools(df, dd):
    """Return True if the hbi_liquidstools column tests pass."""
    raise NotImplementedError

def hbi_abdominalmass(df, dd):
    """Return True if the hbi_abdominalmass column tests pass."""
    raise NotImplementedError

def hbi_complications(df, dd):
    """Return True if the hbi_complications column tests pass."""
    raise NotImplementedError

def hbi_abscess(df, dd):
    """Return True if the hbi_abscess column tests pass."""
    raise NotImplementedError

def hbi_fissure(df, dd):
    """Return True if the hbi_fissure column tests pass."""
    raise NotImplementedError

def hbi_aphthous(df, dd):
    """Return True if the hbi_aphthous column tests pass."""
    raise NotImplementedError

def hbi_arthralgia(df, dd):
    """Return True if the hbi_arthralgia column tests pass."""
    raise NotImplementedError

def hbi_nodosum(df, dd):
    """Return True if the hbi_nodosum column tests pass."""
    raise NotImplementedError

def hbi_fistula(df, dd):
    """Return True if the hbi_fistula column tests pass."""
    raise NotImplementedError

def hbi_gangrenosum(df, dd):
    """Return True if the hbi_gangrenosum column tests pass."""
    raise NotImplementedError

def hbi_uveitis(df, dd):
    """Return True if the hbi_uveitis column tests pass."""
    raise NotImplementedError

def hbi_total(df, dd):
    """Return True if the hbi_total column tests pass."""
    raise NotImplementedError

def sccai_stoolsperday(df, dd):
    """Return True if the sccai_stoolsperday column tests pass."""
    raise NotImplementedError

def sccai_stoolspernight(df, dd):
    """Return True if the sccai_stoolspernight column tests pass."""
    raise NotImplementedError

def sccai_urgency(df, dd):
    """Return True if the sccai_urgency column tests pass."""
    raise NotImplementedError

def sccai_blood(df, dd):
    """Return True if the sccai_blood column tests pass."""
    raise NotImplementedError

def sccai_wellbeing(df, dd):
    """Return True if the sccai_wellbeing column tests pass."""
    raise NotImplementedError

def sccai_eim(df, dd):
    """Return True if the sccai_eim column tests pass."""
    raise NotImplementedError

def sccai_arthralgia(df, dd):
    """Return True if the sccai_arthralgia column tests pass."""
    raise NotImplementedError

def sccai_nodosum(df, dd):
    """Return True if the sccai_nodosum column tests pass."""
    raise NotImplementedError

def sccai_gangrenosum(df, dd):
    """Return True if the sccai_gangrenosum column tests pass."""
    raise NotImplementedError

def sccai_uveitis(df, dd):
    """Return True if the sccai_uveitis column tests pass."""
    raise NotImplementedError

def sccai_total(df, dd):
    """Return True if the sccai_total column tests pass."""
    raise NotImplementedError

def pucai_abdpain(df, dd):
    """Return True if the pucai_abdpain column tests pass."""
    raise NotImplementedError

def pucai_bleeding(df, dd):
    """Return True if the pucai_bleeding column tests pass."""
    raise NotImplementedError

def pucai_consistency(df, dd):
    """Return True if the pucai_consistency column tests pass."""
    raise NotImplementedError

def pucai_frequency(df, dd):
    """Return True if the pucai_frequency column tests pass."""
    raise NotImplementedError

def pucai_nocturnal(df, dd):
    """Return True if the pucai_nocturnal column tests pass."""
    raise NotImplementedError

def pucai_limitation(df, dd):
    """Return True if the pucai_limitation column tests pass."""
    raise NotImplementedError

def pucai_total(df, dd):
    """Return True if the pucai_total column tests pass."""
    raise NotImplementedError

def anymeds(df, dd):
    """Return True if the anymeds column tests pass."""
    raise NotImplementedError

def current_5asa(df, dd):
    """Return True if the current_5asa column tests pass."""
    raise NotImplementedError

def asa_type(df, dd):
    """Return True if the asa_type column tests pass."""
    raise NotImplementedError

def current_antibiotic(df, dd):
    """Return True if the current_antibiotic column tests pass."""
    raise NotImplementedError

def antibiotic_type(df, dd):
    """Return True if the antibiotic_type column tests pass."""
    raise NotImplementedError

def other_antibiotic(df, dd):
    """Return True if the other_antibiotic column tests pass."""
    raise NotImplementedError

def current_probiotic(df, dd):
    """Return True if the current_probiotic column tests pass."""
    raise NotImplementedError

def probiotic_type(df, dd):
    """Return True if the probiotic_type column tests pass."""
    raise NotImplementedError

def other_probiotic(df, dd):
    """Return True if the other_probiotic column tests pass."""
    raise NotImplementedError

def current_glucocorticoid(df, dd):
    """Return True if the current_glucocorticoid column tests pass."""
    raise NotImplementedError

def glucocorticoid_type(df, dd):
    """Return True if the glucocorticoid_type column tests pass."""
    raise NotImplementedError

def current_immunomodulator(df, dd):
    """Return True if the current_immunomodulator column tests pass."""
    raise NotImplementedError

def immunomodulator_type(df, dd):
    """Return True if the immunomodulator_type column tests pass."""
    raise NotImplementedError

def current_antitnf(df, dd):
    """Return True if the current_antitnf column tests pass."""
    raise NotImplementedError

def antitnf_type(df, dd):
    """Return True if the antitnf_type column tests pass."""
    raise NotImplementedError

def current_calcineurin(df, dd):
    """Return True if the current_calcineurin column tests pass."""
    raise NotImplementedError

def calcineurin_type(df, dd):
    """Return True if the calcineurin_type column tests pass."""
    raise NotImplementedError

def current_otherbiologics(df, dd):
    """Return True if the current_otherbiologics column tests pass."""
    raise NotImplementedError

def otherbiologic_type(df, dd):
    """Return True if the otherbiologic_type column tests pass."""
    raise NotImplementedError

def fecaltransplant(df, dd):
    """Return True if the fecaltransplant column tests pass."""
    raise NotImplementedError

def current_othermeds(df, dd):
    """Return True if the current_othermeds column tests pass."""
    raise NotImplementedError

def othermed_type(df, dd):
    """Return True if the othermed_type column tests pass."""
    raise NotImplementedError

def othermed(df, dd):
    """Return True if the othermed column tests pass."""
    raise NotImplementedError

def date_labs(df, dd):
    """Return True if the date_labs column tests pass."""
    raise NotImplementedError

def hct(df, dd):
    """Return True if the hct column tests pass."""
    raise NotImplementedError

def wbc(df, dd):
    """Return True if the wbc column tests pass."""
    raise NotImplementedError

def plt(df, dd):
    """Return True if the plt column tests pass."""
    raise NotImplementedError

def crp(df, dd):
    """Return True if the crp column tests pass."""
    raise NotImplementedError

def esr(df, dd):
    """Return True if the esr column tests pass."""
    raise NotImplementedError

def alb(df, dd):
    """Return True if the alb column tests pass."""
    raise NotImplementedError

def notes(df, dd):
    """Return True if the notes column tests pass."""
    raise NotImplementedError

def ibd_biorepository_sample_database_complete(df, dd):
    """Return True if the ibd_biorepository_sample_database_complete column tests pass."""
    raise NotImplementedError




