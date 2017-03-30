#!/usr/bin/env python
"""Code supporting the information discovery and assimilation of data/file assets."""

# Imports
import os
from pathlib import Path
from collections import defaultdict, namedtuple
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from munch import Munch, munchify

import biorep_etl.errors as e

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"

# Constants
PARSE_FILE_NAME = Munch()

# Classes
Row = namedtuple('Row', ["path_hash","file_name",
                         "directory","batch_code",
                         "file_type","assay_type",
                         "bytes","subject_id"], verbose=False, rename=False)



# Functions
def pathify_assets(FILE_TYPE):
    """Converts the list of path glob patterns in the config file to list of ``Path`` objects.
    
    In place conversion.
    
    Args:
        FILE_TYPE (``dict``-like): key=file type, val=list of path glob patterns
    
    Returns:
        ``None``
    """
    for key in FILE_TYPE.keys():
        paths = []
        for i in FILE_TYPE[key]:
            p = Path(i)
            paths.extend(list(p.parent.glob(p.name)))
            
        FILE_TYPE[key] = paths
    
    
def build_asset_table(asset_conf, pathify=True):
    """Return asset table as ``pd.DataFrame`` built from ``asset_conf`` info.
    
    Column Discriptions:
        - path_hash (`int`)
        - file_name (`str`)
        - directory (`str`)
        - batch_code (`Category`)
            - Regeneron1, Merck1, Merck2, etc
        - file_type (`Category`)
            - BAM, VCF, GVCF, FASTQ, etc
        - assay_type (`Category`)
            - WES, WGS, RNAseq, etc
        - bytes (`int`)
        - subject_id (`str`)
    
    Args:
        asset_conf (``dict``-like): configuration tree built from asset_intake configuration file.
        pathify (``bool``): whether or not to run ``pathify_assets()`` on the paths in ``asset_conf``
    
    Returns:
        ``pd.DataFrame``
    """
    dtypes = {"path_hash": np.int64,
              "file_name": str,
              "directory": str,
              "batch_code": "category",
              "file_type": "category",
              "assay_type": "category",
              "bytes": np.int64,
              "subject_id": str,
              }
    
    if pathify:
        for batch in asset_conf.BATCHES.values():
            pathify_assets(FILE_TYPE=batch.FILE_TYPE)
    
    rows = []
    for batch_name, batch in asset_conf.BATCHES.items():
        for ftype, paths in batch.FILE_TYPE.items():
            for path in paths:

                path_hash = hash(str(path))
                file_name = path.name
                directory = str(path.parent)
                batch_code = batch_name
                file_type = ftype
                assay_type = batch.ASSAY_TYPE
                bytes = path.stat().st_size
                subject_id = path.stem
                
                rows.append(Row(path_hash=path_hash,
                                file_name=file_name,
                                directory=directory,
                                batch_code=batch_code,
                                file_type=file_type,
                                assay_type=assay_type,
                                bytes=bytes,
                                subject_id=subject_id
                                )
                            )
    assets = pd.DataFrame(data=rows, index=None, columns=None, dtype=None, copy=False).astype(dtype=dtypes, copy=True, raise_on_error=True)

    return assets
                


def subject_from_regeneron1_fname(fname):
    """Parse a file name from the REGENERON1 data batch to as close to a subject_id as possible.
    
    Args:
        fname (``str``): a file name.
        
    Returns:
        ``str``
    """
    return fname.split('_')[1].rstrip('P').rstrip('M').rstrip('D').rstrip('F')

PARSE_FILE_NAME.REGENERON1 = subject_from_regeneron1_fname
    