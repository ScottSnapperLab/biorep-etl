#!/usr/bin/env python
"""Provide error classes for biorep-etl."""

# Imports


# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"




class Biorep-etlError(Exception):

    """Base error class for biorep-etl."""


class ValidationError(Biorep-etlError):

    """Raise when a validation/sanity check comes back with unexpected value."""
