#!/usr/bin/env python
"""Provide parsing functions specific to values found in BCH biorepository fields."""

# Imports
from logzero import logger as log

import attr

from biorep_etl.data import validation as v
import biorep_etl.errors as e


@attr.s
class BiorepIDNumber(object):
    """Represent the original and digested information encoded in a biorepidnumber."""

    __valid_delims = ['.', '-']
    __invalid_relations = ["1"]
    __relation_map = {None: "proband",
                      "0": "proband",
                      "2": "mother",
                      "3": "father",
                      "4": "sibling",
                      "5": "sibling",
                      "6": "sibling",
                      "7": "sibling",
                      }

    original = attr.ib(convert=str)

    proband_id = attr.ib(init=False,
                         validator=attr.validators.instance_of(str))

    relation_to_proband = attr.ib(init=False,
                                  validator=attr.validators.instance_of(str))

    delim = attr.ib(default='.',
                    validator=attr.validators.in_(__valid_delims))

    def __attrs_post_init__(self):
        """Perform custom __init__ actions after the attrs init func."""
        try:
            proband_id, relation_to_proband = self.original.split(self.delim)
            self.proband_id = proband_id
            self.relation_to_proband = [relation_to_proband]
        except ValueError as exc:
            if "not enough values to unpack" in exc.args[0]:
                self.proband_id = self.original
                self.relation_to_proband = None
            else:
                raise

        attr.validate(self)

    @relation_to_proband.validator
    def __relations_not_in(self, attribute, value):
        """Validate that ``relation_to_proband`` is not in ``value``."""
        if value in self.__invalid_relations:
            msg = "self.relation_to_proband ({value}) must not be in {invalids}.".format(value=self.relation_to_proband,
                                                                                         invalids=self.__invalid_relations)
            log.error(msg)
            raise e.ValidationError(msg)
