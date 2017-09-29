#!/usr/bin/env python
"""Provide code supporting the running and automating of Snakemake rules."""

# Imports
from snaketools.snaketools import apply_template
from snaketools.snaketools import pathify_by_key_ends
from snaketools.snaketools import SnakeRun
from snaketools.snaketools import SnakeRule
from snaketools.snaketools import recode_graph

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"


__all__ = ["apply_template", "pathify_by_key_ends", "SnakeRun", "SnakeRule", "recode_graph"]
