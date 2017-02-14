#!/usr/bin/env python
"""Provide pyparsing parsers for the project."""

# Imports
from lxml import etree
from lxml import objectify

import pyparsing as p

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"

# Functions
def build_choices_parser():
    """Return a pyparsing object built to digest choice label defintions."""
    choice_delim = p.Literal('|').suppress()
    value_delim = p.Literal(',').suppress()
    
    choice_int = p.Word('-' + p.nums) + p.FollowedBy(',')
    choice_txt = p.Word(p.printables + ' ', excludeChars='|').setParseAction(lambda toks: toks[0].rstrip())
    
    choice = p.Group(p.Optional(choice_delim) + choice_int + value_delim + choice_txt + p.Optional(choice_delim))
    
    return p.OneOrMore(choice, stopOn=None)

def build_branch_logic_parser():
    """Return a pyparsing object built to digest redcap branching logic notationself."""
    
    # lay out the grammar
    
    ## misc
    lbrack = p.Literal("[").suppress()
    rbrack = p.Literal("]").suppress()
    lparen = p.Literal("(").suppress()
    rparen = p.Literal(")").suppress()
    anyquote = p.oneOf("""" '""").suppress()
    empty_quotes = p.Literal("''") | p.Literal('""')

    ## variables
    checkbox_val = p.Optional('('+p.Word(p.nums)+')')
    variable_simple = p.Combine(lbrack+p.Word(p.alphanums + "_")+checkbox_val+rbrack)
    variable = p.Group(p.OneOrMore(variable_simple)).setResultsName('variable')

    ## operators
    operator = p.oneOf("<> >= <= > < =")

    ## logic gates
    AND = p.Literal("and")
    OR = p.Literal("or")
    logic_gate = p.Group(AND | OR).setResultsName('logic_gate')

    ## comparison vals
    comparison_val = p.Group(p.Word(p.alphanums) | (anyquote + p.Word(p.alphanums) + anyquote) | empty_quotes).setResultsName('comparison_val')

    # ## branch expression
    # # expression =
    # expression = (variable) + logic_gate + (comparison_val)


    ## atom = expression | variable | comparison_val
    atom = p.Group(variable | comparison_val).setResultsName('atom')
    
    
    # define the operator precedence
    parser = p.operatorPrecedence( atom,
                                  [(operator, 2, p.opAssoc.LEFT),
                                   (logic_gate, 2, p.opAssoc.RIGHT),]
                                  )
    return parser


# Classes
class BranchLogicParser(object):
    """Manages the initiation, execution, and results navigation of a branch-logic-parser."""
    # build this once since it should be stateless to reduce cost of instantiation.
    parser = build_branch_logic_parser()
    
    def __init__(self):
        """Initialize object.
        
        No args taken. Note: The self.parser is a CLASS-level attribute.
        """
        self.result = None
        self.result_xml = None
        self.result_obj = None
        self.variables = None
        self.comparison_vals = None
    
    def parse_string(self, string):
        """Parse ``string`` and store result in ``self.result``.
        
        Args:
            string (str): string containing branching logic notation.
        
        Returns:
            None
        """
        
        self.result = self.parser.parseString(string)
        self.result_xml = self.result.asXML()
        self.result_obj = objectify.fromstring(self.result_xml)
        self.variables = [o["ITEM"][:] for o in self.result_obj.xpath('//variable')]
        self.comparison_vals = [o["ITEM"][:] for o in self.result_obj.xpath('//comparison_val')]
        

