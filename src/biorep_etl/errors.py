#!/usr/bin/env python
"""Provide error classes for biorep_etl."""

# Imports


# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"




class BiorepEtlError(Exception):

    """Base error class for biorep_etl."""


class ValidationError(BiorepEtlError):

    """Raise when a validation/sanity check comes back with unexpected value."""
