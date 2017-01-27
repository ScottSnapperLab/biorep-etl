#!/usr/bin/env python
"""Provide code to build a DAG from the CRF branching-logic information."""

# Imports
from collections import namedtuple

import pandas as pd
import numpy as np

import networkx as nx

from munch import Munch

from biorep_etl.data.parsers import BranchLogicParser
import biorep_etl.errors as e

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"


# Constants
EdgeLabel = namedtuple(typename='EdgeLabel', field_names=["n1", "n2", "label"], verbose=False, rename=False)

# Functions
def make_edges(data_dict):
    """Return df with just columns that represent edges."""
    rename = {'Variable / Field Name':'field_name',
              'Branching Logic (Show field only if...)':'branch_logic'}
    
    return data_dict.reset_index()[['Variable / Field Name','Branching Logic (Show field only if...)']].rename(columns=rename)


def make_top_level_nodes_and_others(edges_raw):
    """Return dict of `top_level_nodes`,`lower_edges_raw`."""
    d = Munch()
    
    tln = edges_raw.branch_logic.apply(lambda v: pd.isnull(v))
    
    d.top_level_nodes = edges_raw.field_name[tln]
    d.lower_edges_raw = edges_raw[~tln]
    
    return  d
    


def parse_branch_logic(logic, n2):
    """Yield ``EdgeLabel`` namedtuples derived from each branch comparison encoded in ``logic``.
    
    logic (str): the redcap table-row's branch_logic column string.
    n2 (str): the redcap table-row's field_name.
    
    Yields:
        EdgeLabel: namedtuples ("n1", "n2", "label")
    """
    parser = BranchLogicParser()
    parser.parse_string(string=logic)
    
    for v in parser.variables:
        yield EdgeLabel(":".join(str(i) for i in v), n2, "") # TODO: add actual labels


def make_lower_edges(lower_edges_raw):
    """Yield ``EdgeLabel`` namedtuples ("n1", "n2", "label") for the lower edges based on branch logic."""
    field_name, branch_logic = 0,1
    
    for row in lower_edges_raw.values:
        
        yield parse_branch_logic(logic=row[branch_logic], n2=row[field_name])
        
            
        

def add_lower_edges(g,lower_edges_raw):
    """Add lower level nodes with edge labels."""
    for lower_edges in make_lower_edges(lower_edges_raw):
        for e in lower_edges:
            g.add_edge(u=e.n1, v=e.n2, label=e.label)
        
    
def add_top_level_edges(g,top_level_nodes):
    """Add top level nodes top graph."""
    l = len(top_level_nodes)
    
    tle = zip(['start']*l,
              top_level_nodes)
    
    g.add_edges_from(ebunch=tle)


def build_dag(data_dict):
    """Return the DAG as decoded from the data_dict dataframe."""
    g = nx.DiGraph()
    
    edges_raw = make_edges(data_dict)
    nodes_raw = make_top_level_nodes_and_others(edges_raw)
    
    add_top_level_edges(g, top_level_nodes=nodes_raw.top_level_nodes)
    # g.add_nodes_from(nodes=nodes_raw.top_level_nodes)
    add_lower_edges(g,lower_edges_raw=nodes_raw.lower_edges_raw)
    
    return g