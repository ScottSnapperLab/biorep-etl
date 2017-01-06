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


import biorep_etl.data.field_definitions.redcap_dump as rcd
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
    dtypes = {'samplenumber': np.object,
              'biorepidnumber': np.object,}
    
    return pd.read_csv(data_, dtype=dtypes, index_col=0)
    


def init_load_data_and_labels(data_, data_dict_):
    """Return Munch obj containing the loaded and verified data dump and the field_map."""
    loaded = Munch()
    
    data_dict = load_data_dict(data_dict_=data_dict_)
    required = rcd.req_cols(dd=data_dict)
    
    loaded.data = load_redcap_dump(data_=data_).pipe(ck.none_missing,
                                                     columns=required)
    
    
    loaded.field_map = make_field_map(data_dict=data_dict)
    
    

    
    return loaded
    
    

def process_where_stored(row):
    """Process where_stored."""
    id_map = {
        "where_stored___1": "Stored in the biorepository",
        "where_stored___2": "Given to Camilla",
        "where_stored___3": "Shipped to the Netherlands",
        "where_stored___4": "Given to Pfizer",
        "where_stored___7": "Shipped to Stanford",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    if row['where_stored___5'].upper() == "CHECKED":
        out.append("other:{kind}".format(kind=row['other_shipped']))

    if row['where_stored___6'].upper() == "CHECKED":
        out.append("labmember:{kind}".format(kind=row['labmember']))

    return [str(x) for x in out]

def process_surgicallocations(row):
    """Process surgicallocations."""
    id_map = {
        "surgicallocations___1": "Ileum inflamed",
        "surgicallocations___9": "Ileum uninflamed",
        "surgicallocations___2": "Cecum inflamed",
        "surgicallocations___10": "Cecum uninflamed",
        "surgicallocations___3": "Right colon inflamed",
        "surgicallocations___11": "Right colon uninflamed",
        "surgicallocations___21": "Hepatic flexure inflamed",
        "surgicallocations___22": "Hepatic flexure uninflamed",
        "surgicallocations___4": "Transverse colon inflamed",
        "surgicallocations___12": "Transverse colon uninflamed",
        "surgicallocations___23": "Splenic flexure inflamed",
        "surgicallocations___24": "Splenic flexure uninflamed",
        "surgicallocations___5": "Left colon inflamed",
        "surgicallocations___13": "Left colon uninflamed",
        "surgicallocations___6": "Sigmoid colon inflamed",
        "surgicallocations___14": "Sigmoid colon uninflamed",
        "surgicallocations___7": "Rectum inflamed",
        "surgicallocations___15": "Rectum uninflamed",
        "surgicallocations___8": "Appendix inflamed",
        "surgicallocations___16": "Appendix uninflamed",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]

def process_stoolmedia(row):
    """Process stoolmedia."""
    id_map = {
        "stoolmedia___1": "Freeze",
        "stoolmedia___2": "RNA later",
        "stoolmedia___3": "Ethanol",
        "stoolmedia___4": "DNA Genotek",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]

def process_probiotic_type(row):
    """Process probiotic_type."""
    id_map = {
         "probiotic_type___1": "VSL3",
         "probiotic_type___2": "LGG (Culturelle)",
         "probiotic_type___3": "Florastor",
        }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    if row['probiotic_type___4'].upper() == "CHECKED":
        out.append("other:{kind}".format(kind=row['other_probiotic']))

    return [str(x) for x in out]


def process_othermed_type(row):
    """Process othermed_type."""
    id_map = {
        "othermed_type___2": "Thalidomide",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    if row['othermed_type___3'].upper() == "CHECKED":
        out.append("other:{kind}".format(kind=row['othermed']))

    return [str(x) for x in out]


def process_otherbiologic_type(row):
    """Process otherbiologic_type."""
    id_map = {
        "otherbiologic_type___1": "Natalizumab (Tysabri)",
        "otherbiologic_type___2": "Ustekinumab (Stelara)",
        "otherbiologic_type___3": "Vedolizumab (Entyvio)",

    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]

def process_oral_perianal(row):
    """Process oral_perianal."""
    id_map = {
        "oral_perianal___1": "Oral",
        "oral_perianal___2": "Perianal",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]

def process_immunomodulator_type(row):
    """Process immunomodulator_type."""
    id_map = {
        "immunomodulator_type___1": "6-mercaptopurine (6-MP)",
        "immunomodulator_type___2": "Azathioprine (Imuran)",
        "immunomodulator_type___3": "Methotrexate (Folex, Rheumatrex, Mexate)",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]


def process_glucocorticoid_type(row):
    """Process glucocorticoid_type."""
    id_map = {
        "glucocorticoid_type___1": "Budesonide (Entocort)",
        "glucocorticoid_type___2": "Hydrocortisone enema (Cortenema, Colocort, Cortifoam)",
        "glucocorticoid_type___3": "Methylprednisolone (Solumedrol)",
        "glucocorticoid_type___4": "Prednisone, prednisolone (Prelone)",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]


def process_familymember_ibd(row):
    """Process familymember_ibd."""
    id_map = {
        "familymember_ibd___1": "Mother",
        "familymember_ibd___2": "Father",
        "familymember_ibd___3": "Full brother",
        "familymember_ibd___4": "Full sister",
        "familymember_ibd___5": "2nd degree relative",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    if row['familymember_ibd___6'].upper() == "CHECKED":
        out.append("other:{kind}".format(kind=row['other_relative']))

    return [str(x) for x in out]

def process_calcineurin_type(row):
    """Process calcineurin_type."""
    id_map = {
        "calcineurin_type___1": "Cyclosporine",
        "calcineurin_type___2": "Tacrolimus",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]

def process_bxlocation(row):
    """Process bxlocation."""
    id_map = {
        "bxlocation___1": "Stomach Antrum inflamed",
        "bxlocation___11": "Stomach Antrum uninflamed",
        "bxlocation___2": "Stomach Corpus inflamed",
        "bxlocation___12": "Stomach Corpus uninflamed",
        "bxlocation___3": "Duodenum inflamed",
        "bxlocation___13": "Duodenum uninflamed",
        "bxlocation___4": "Terminal Ileum inflamed",
        "bxlocation___14": "Terminal Ileum uninflamed",
        "bxlocation___5": "Cecum inflamed",
        "bxlocation___15": "Cecum uninflamed",
        "bxlocation___6": "Right colon inflamed",
        "bxlocation___16": "Right colon uninflamed",
        "bxlocation___21": "Hepatic flexure inflamed",
        "bxlocation___22": "Hepatic flexure uninflamed",
        "bxlocation___7": "Transverse colon inflamed",
        "bxlocation___17": "Transverse colon uninflamed",
        "bxlocation___23": "Splenic flexure inflamed",
        "bxlocation___24": "Splenic flexure uninflamed",
        "bxlocation___8": "Left colon inflamed",
        "bxlocation___18": "Left colon uninflamed",
        "bxlocation___9": "Sigmoid colon inflamed",
        "bxlocation___19": "Sigmoid colon uninflamed",
        "bxlocation___20": "Rectum inflamed",
        "bxlocation___10": "Rectum uninflamed",
        "bxlocation___25": "Pouch inflamed",
        "bxlocation___26": "Pouch uninflamed",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]

def process_blood_samples(row):
    """Process blood_samples."""
    id_map = {
        "blood_samples___5": "DNA",
        "blood_samples___3": "PAX gene",
        "blood_samples___2": "PBMCs",
        "blood_samples___1": "Serum",
        "blood_samples___6": "Sodium heparin tube",
        "blood_samples___4": "Whole blood",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]


def process_asa_type(row):
    """Process asa_type."""
    id_map = {
        "asa_type___1": "Mesalamine (Asacol, Canasa, Pentasa, Rowasa, Apriso, Lialda)",
        "asa_type___2": "Balsalazide (Colazal)",
        "asa_type___3": "Sulfasalazine (Azulfidine)",
        "asa_type___4": "Olsalazine (Dipentum)",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]


def process_antitnf_type(row):
    """Process antitnf_type."""
    id_map = {
        "antitnf_type___1": "Infliximab (Remicade)",
        "antitnf_type___2": "Adalimumab (Humira)",
        "antitnf_type___3": "Certolizumab pegol (Cimzia)",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    return [str(x) for x in out]

def process_antibiotic_type(row):
    """Process antibiotic_type."""
    id_map = {
        "antibiotic_type___1": "Amoxicillin (Amoxil), Amoxicillin/Clavulanate (Augmentin)",
        "antibiotic_type___2": "Ciprofloxacin (Cipro)",
        "antibiotic_type___3": "Metronidazole (Flagyl)",
        "antibiotic_type___4": "Rifaximin (Xifaxin)",
        "antibiotic_type___5": "Trimethoprim/Sulfamethoxazole (Bactrim)",
        "antibiotic_type___6": "Vancomycin",
    }

    out = []

    for col in id_map.keys():
        if row[col].upper() == "CHECKED":
            out.append(id_map[col])

    if row['antibiotic_type___7'].upper() == "CHECKED":
        out.append("other:{kind}".format(kind=row['other_antibiotic']))

    return [str(x) for x in out]



def process_all(df):
    """Process all of the columns needing to be dedummied."""
    funcs = {
        "where_stored": process_where_stored,
        "surgicallocations": process_surgicallocations,
        "stoolmedia": process_stoolmedia,
        "probiotic_type": process_probiotic_type,
        "othermed_type": process_othermed_type,
        "otherbiologic_type": process_otherbiologic_type,
        "oral_perianal": process_oral_perianal,
        "immunomodulator_type": process_immunomodulator_type,
        "glucocorticoid_type": process_glucocorticoid_type,
        "familymember_ibd": process_familymember_ibd,
        "calcineurin_type": process_calcineurin_type,
        "bxlocation": process_bxlocation,
        "blood_samples": process_blood_samples,
        "asa_type": process_asa_type,
        "antitnf_type": process_antitnf_type,
        "antibiotic_type": process_antibiotic_type,
    }

    df_ = df.copy()

    for name, func in funcs.items():
        df_[name] = df_.apply(func, axis=1)

    return df_




def drop_them(df):
    """Drop the columns that we just dedummied"""
    drop_me = (
        "where_stored___1",
        "where_stored___2",
        "where_stored___3",
        "where_stored___4",
        "where_stored___7",
        "where_stored___5",
        "where_stored___6",
        "surgicallocations___1",
        "surgicallocations___9",
        "surgicallocations___2",
        "surgicallocations___10",
        "surgicallocations___3",
        "surgicallocations___11",
        "surgicallocations___21",
        "surgicallocations___22",
        "surgicallocations___4",
        "surgicallocations___12",
        "surgicallocations___23",
        "surgicallocations___24",
        "surgicallocations___5",
        "surgicallocations___13",
        "surgicallocations___6",
        "surgicallocations___14",
        "surgicallocations___7",
        "surgicallocations___15",
        "surgicallocations___8",
        "surgicallocations___16",
        "stoolmedia___1",
        "stoolmedia___2",
        "stoolmedia___3",
        "stoolmedia___4",
        "probiotic_type___1",
        "probiotic_type___2",
        "probiotic_type___3",
        "probiotic_type___4",
        "othermed_type___2",
        "othermed_type___3",
        "otherbiologic_type___1",
        "otherbiologic_type___2",
        "otherbiologic_type___3",
        "oral_perianal___1",
        "oral_perianal___2",
        "immunomodulator_type___1",
        "immunomodulator_type___2",
        "immunomodulator_type___3",
        "glucocorticoid_type___1",
        "glucocorticoid_type___2",
        "glucocorticoid_type___3",
        "glucocorticoid_type___4",
        "familymember_ibd___1",
        "familymember_ibd___2",
        "familymember_ibd___3",
        "familymember_ibd___4",
        "familymember_ibd___5",
        "familymember_ibd___6",
        "calcineurin_type___1",
        "calcineurin_type___2",
        "bxlocation___1",
        "bxlocation___11",
        "bxlocation___2",
        "bxlocation___12",
        "bxlocation___3",
        "bxlocation___13",
        "bxlocation___4",
        "bxlocation___14",
        "bxlocation___5",
        "bxlocation___15",
        "bxlocation___6",
        "bxlocation___16",
        "bxlocation___21",
        "bxlocation___22",
        "bxlocation___7",
        "bxlocation___17",
        "bxlocation___23",
        "bxlocation___24",
        "bxlocation___8",
        "bxlocation___18",
        "bxlocation___9",
        "bxlocation___19",
        "bxlocation___20",
        "bxlocation___10",
        "bxlocation___25",
        "bxlocation___26",
        "blood_samples___5",
        "blood_samples___3",
        "blood_samples___2",
        "blood_samples___1",
        "blood_samples___6",
        "blood_samples___4",
        "asa_type___1",
        "asa_type___2",
        "asa_type___3",
        "asa_type___4",
        "antitnf_type___1",
        "antitnf_type___2",
        "antitnf_type___3",
        "antibiotic_type___1",
        "antibiotic_type___2",
        "antibiotic_type___3",
        "antibiotic_type___4",
        "antibiotic_type___5",
        "antibiotic_type___6",
        "antibiotic_type___7",
        )

    return df.copy().drop(drop_me)




















