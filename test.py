import os
import math
import json
import argparse
import types
import logging

import yaml
import spacy

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# Fill textL[] with a strings containing the text of each input document
def test( ):
    textL     = []
    filenameL = []
    for filename in os.listdir("test docs"):
        filenameL.append(filename)
        with open( os.path.join("test docs",filename), "r" ) as f:
            textL.append( f.read() )
    #print(textL[1])
    print(filenameL[0])

    # Process each document through the spacy pipeline
    tok_docL = zip(filenameL,nlp.pipe(textL))
    print(list(tok_docL))

if __name__ == "__main__":
     # Load the spacy pipeline based on the specified language model
    nlp = spacy.load('en_core_web_sm')
    
    test()