#!/usr/bin/env python
"""Provide code to validate loaded data tables."""

# Imports


# Functions
def is_in(x, y):
    """Return ``True`` if ``x`` is in ``y``, ``False`` otherwise."""
    if x in y:
        return True
    else:
        return False


def not_in(x, y):
    """Return ``True`` if ``x`` is NOT in ``y``, ``False`` otherwise."""
    if x not in y:
        return True
    else:
        return False
