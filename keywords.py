""" 
keyword.py : Find the keywords in a corpus of documents.
Usage: See python keyword.py --help.
"""
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

class DocTerm:
    """ DocTerm represents attributes of each term in the source spacy document.

    token_indexL -- A list of token indexes into the source spacy document.
    termN        -- Count of times this term occurs in the source spacy documnent
    """
    def __init__( self ):
        self.token_indexL = [] 
        self.termN = 0

    def add_term_instance( self, token_indexL ):
        """ Add the tokens associated with other instances of this term to the term index list."""
        self.token_indexL += token_indexL
        self.termN += 1
        
class CorpusTerm:
    """
    CorpusTerm represents attributes of terms in the global corpus vocabulary.
    
    docN   -- Count of documents this term occurs in.
    tfidf  -- TF/IDF score of this term.
    tfidfN -- average of tfidf scores for all ????
    termN  -- Total number of times this term occurs in the corpus
    """
    def __init__( self ):
        self.docN   = 0
        self.tfidf  = 0
        self.tfidfN = 0
        self.termN  = 0

    def add_corpus_instance(self):
        self.docN += 1

class Doc:
    """
    Doc represents a source document.

    doc      -- spacy document object
    filename -- The name of the file this document was created from.
    wordD    -- { term:DocTerm } Vocabulary for this document.
    """
    def __init__(self,doc,filename,wordD):
        self.doc      = doc                 # need this because this is how i am going to get back to sentence containing term
        self.filename = filename
        self.wordD    = wordD

    def calc_tf_idf( self, corpusDocN, corpusD ):
        """ Calculate TF/IDF on each word in this document and update the
        per word aggregate TF/IDF CorpusD{}

        corpusDocN  -- number of documents in corpus
        corpusD     -- global vocabulary (dictionary)

        """
        
        # for each word in this document
        # getting back the key : value from wordD
        for word,docTerm in self.wordD.items():

            # compute the term frequency
            # len(self.doc) includes stop words because it is the space_doc
            tf = docTerm.termN / len(self.doc)

            # compute the inverse document frequency
            idf   = math.log(corpusDocN / (1 + corpusD[word].docN))

            # calc the sum of TF/IDF for this term across all docs
            corpusD[word].tfidf  += tf*idf

            # track the number of times this term occurs in the corpus
            # tfidfN ends up being the same as docN
            corpusD[word].tfidfN += 1

            # count of occurances of this term in this document
            # need this for output
            corpusD[word].termN += docTerm.termN
            
        return corpusD

    def gen_per_doc_output( self, cfg, tfidfL ):
        """ Genrate the OutputRecords associated with this document
        when the output set to a fixed number of terms per document.
        """
        outL = []
        
        logging.info(self.filename)
        
        for i,(wordL,tfidf,termNL) in enumerate(tfidfL):

            wL = [ word for word in wordL if word in self.wordD ]

            if len(wL) > 0:
                logging.info("%3i %5.4f %s",i,tfidf,wL)

                out_recd = OutputRecord( wL, tfidf, termNL )

                self.find_sentences_in_doc( out_recd )
                
                outL.append(out_recd)
                
                if len(outL) == cfg.perDocKeyWordCount:
                    break
                
        return outL

    def find_sentences_in_doc( self, out_recd ):
        """ Given an output record find all the sentences in this document containing out_recd.term"""

        # store the filename/sentences containing the keyword
        for term in out_recd.termL:
            if term in self.wordD:
                for tokIdx in self.wordD[ term ].token_indexL:
                    logging.info("%i %s",tokIdx, self.filename)
                    # self.doc is the spacy doc so that we can get back to sentence or span by giving index of token
                    out_recd.insert_sentence(self.filename,self.doc[tokIdx].sent)
                

class OutputRecord:
    """
    OutputRecord represents a keyword selected to be written to the output.

    termL     -- keywords text
    tfidf     -- TF/IDF score for this term
    termNL    -- Total occurences of each term in the corpus.
    filenameL -- List of files containing this keyword.
    sentenceL -- List of sentences containing this keyword.
    """
    def __init__( self, termL, tfidf, termNL ):
        self.termL     = termL
        self.tfidf     = tfidf
        self.termNL    = termNL
        self.filenameL = []
        self.sentenceL = []

    def insert_sentence( self, filename, sentence ):
        if filename not in self.filenameL:
            self.filenameL.append(filename)

        if str(sentence) not in self.sentenceL:
            self.sentenceL.append(str(sentence))


def doc_to_words( cfg, doc ):
    """ This function builds a list of single and multi-token terms based on
    the token attributes assigned by the spacy pipeline.
    It then returns a dictionary associating each term with the indexes
    into the document token list where the term originates.
    These token indexes will be needed later to locate the sentences in which
    the terms occur.
    """
    
    wordL = [] # ( term, token_indexL )

    # for each token in the doc
    for i,tok in enumerate(doc):

        skipFl = tok.is_currency or tok.is_punct or tok.is_stop or tok.is_space or tok.like_url or tok.like_email or tok.like_num
        
        # if this token is not part of a named entity
        if not tok.ent_type_:

            # skip tokens that were found to be: currency symbols, punctuation, stop words, URLs, email addresses, or numeric
            # tok.lemma is the lemmitization of the word
            if not (cfg.useNamedEntityTokensOnlyFl or skipFl):
                wordL.append( (tok.lemma_,[i]))

        # if this token part of a named entity
        else:
                
            # if this named entity token is not in the 'drop' list
            if tok.ent_type_ in cfg.nerTypeIncludeL:

                # building multi word enties
                text = "" if skipFl else " " + tok.text

                # if this is the first token in a named entity term
                # found lemmitization does not work as well for named entities
                if tok.ent_iob == 3:
                    wordL.append( (text.strip(),[i]) )
                    
                # if this token is part of the previous token (i.e. part of a multi-token term)
                elif tok.ent_iob == 1:
                    # append the token text to the previous term 
                    wordL[-1] = (wordL[-1][0] + text, wordL[-1][1] )

                else:
                    assert 0
    
    # Convert wordL into a dictionary
    # Creating a dictionary of unique terms (e.g., keys have to be unique in dictionary)
    termD = {} # { term: DocTerm }
    
    for word,tokIdxL in wordL:
        if word not in termD:
            termD[word] = DocTerm()
            
        termD[word].add_term_instance( tokIdxL )
                
    return termD

def merge_sub_string( tfidfL, maxN ):
    """ 
    Given the 'maxN' highest scoring key words, search for keywords
    in this set which are substring of other key words in this set.
    Merge two terms into the higher scoring key word and remove
    the lower scoring key word.

    N squared - bad
    """

    del_indexL = []
    for i,(termL,i_score,termNL) in enumerate(tfidfL[0:maxN]):

        for term in termL:
            
            for j,(wordL,j_score,wordNL) in enumerate(tfidfL[0:maxN]):

                for word in wordL:
                
                    if len(term) < len(word) and term in word:
                        logging.info("merge:%s -> %s",term,word)
                        
                        tfidfL[j] = (wordL + [term],(i_score+j_score)/2,wordNL+termNL)

                        del_indexL.append(i)
                        
                        break

    # delete the merged terms
    for i in sorted(del_indexL,reverse=True):
        del tfidfL[i]
        
    return tfidfL

def generate_per_doc_output( cfg, docL, tfidfL ):
    """ Genrate a list of OutputRecords associated 
    where the output set to cfg.perDocKeyWordCount terms per document.
    (i.e. cfg.usePerDocWordCount == True)
    """

    outL = []
    for doc in docL:
        outL += doc.gen_per_doc_output(cfg,tfidfL)

    return outL

def generate_corpus_output( cfg, docL, tfidfL ):
    """ Generate a list of OutputRecords where the number of key words
    is limited to the cfg.corpusKeywordCount highest scoring terms.
    (i.e. cfg.usePerDocWordCount == False)
    """

    outL = []
    
    # for the cfg.corpusKeyWordCount highest scoring keywords
    for i,(wordL,tfidf,termNL) in enumerate(tfidfL[0:min(cfg.corpusKeyWordCount,len(tfidfL))]):

        out_recd = OutputRecord(wordL,tfidf,termNL)

        logging.info("%i %f %s",i,tfidf,wordL)
        
        # for each document 
        for doc in docL:
            doc.find_sentences_in_doc(out_recd)

        outL.append(out_recd)

    return outL


def find_sub_string_indexes( term, sentence ):
    """ Return the starting index of each location in 'sentence' matching 'term'.
    This function is only used for HTML bolding.
    """
    idxL = []
    i = 0
    while i < len(sentence):

        if sentence.startswith( term, i):
            idxL.append((i,len(term)))
            i += len(term)
        else:
            i += 1

    return idxL
            
    
def insert_bold_html_tags( termL, sentenceL ):
    """
    Given a list of terms and a list of sentences insert HTML bold tags
    aroundd all the terms in the sentences.
    """
    sentL = []
    for sentence in sentenceL:
        idxL = []
        for term in termL:
            idxL = find_sub_string_indexes(term,sentence)

        s = ""
        i = 0
        for j,n in idxL:
            s += sentence[i:j]
            s += "<b>" + sentence[j:j+n] + "</b>"
            i  = j+n

        s += sentence[i:]
        sentL.append(s)
        
    return sentL
        
            
    
def write_html_output( outL, htmlFn ):
    """
    Generate output in HTML form.
    """

    textTop = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>NLP Coding Task by Jo Frabetti</title>
        <link rel="stylesheet" href="styles.css">
        <link rel="shortcut icon" href="favicon.jpg"/>
      </head>
    <body>
        <h1>NLP Coding Task </h1>
        <h2>by Jo Frabetti</h2>
        <br>
        <p><strong>Task:</strong> Produce a list of the <strong><i>most frequent interesting</i></strong> words from a corpus of similar documents. </p>
        <p><strong>Result:</strong> For the purpose of this assignment, interesting is defined using the statistical measure <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf"><i>term frequency-inverse document frequency (TF-IDF)</i></a> which offsets raw counts of the number of times a word appears in a document by the number of documents in the corpus that contain the word. <p> 
        <div>
    """
    textBottom = """
        </div>
      </body>
    </html>
    """

    t = "<table id='customers'>"
    t += "<colgroup><col span='1' style='width: 20%;'><col span='1' style='width: 5%;'><col span='1' style='width: 75%;'></colgroup><tr><th>Word (Total Occurances)</th><th>Document(s)</th><th>Sentences Containing the Word</th></tr>"
                
    for out_recd in outL:
        t += "<tr>"

        sentenceL = insert_bold_html_tags( out_recd.termL, out_recd.sentenceL )

        sentText = "<br><br>".join(sentenceL)
        fnText   = "<br><br>".join(out_recd.filenameL)        
        termText = "<br><br>".join([ "%s (%i)" % (term,termN) for term,termN in zip(out_recd.termL,out_recd.termNL) ])

        for v in [termText,fnText,sentText]:
            t +="<td>{}</td>".format(v)
                
        t += "</tr>"
    
    t += "</table>"
    
    with open(htmlFn,"w") as f:
        f.write(textTop + t  + textBottom )
        
# Code from Spacy website to customize tokenizer
def use_infix_pattern_tokenizer( nlp ):
    """
    Install a custom infix pattern tokenizer to correctly handle hyphenated terms.
    """
    # modify tokenizer infix patterns
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # EDIT: commented out regex that splits on hyphens between letters:
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    return nlp

def main( cfg, in_dir  ):
    """
    Main entry point for the program.
    
    cfg   -- Configuration record as returned from parse_yaml_cfg().
    in_dir -- The directory containing the corpus of text files.
    """

    if not os.path.isdir(in_dir):
        logging.error("The input directory '%s' does not exist",in_dir)
        return None

    # Load the spacy pipeline based on the specified language model
    nlp = spacy.load(cfg.langModelName)

    # Use the infix pattern tokenizer
    if cfg.useInfixPatternTokenizerFl:
        nlp = use_infix_pattern_tokenizer(nlp)

    # Fill textL[] with a strings containing the text of each input document
    # we need to associate the filename with the document
    textL     = []
    filenameL = []
    for filename in os.listdir(in_dir):
        filenameL.append(filename)
        with open( os.path.join(in_dir,filename), "r" ) as f:
            textL.append( f.read() )

    # Process each document through the spacy pipeline
    # zip creates a tuple, but nlp.pipe(input) needs text input
    tok_docL = zip(filenameL,nlp.pipe(textL))

    # Create list that contains document name and DocTerm dictionary
    docL    = []  # [ [ Doc ]            :   one element per document
    corpusD = {}  # [ word:CorpusTerm ]  :   one element per unique words in the corpus
    for filename,spacy_doc in tok_docL:

        # Create a dictionary { word:DocTerm } for each document.
        wordD = doc_to_words(cfg,spacy_doc)

        # Load the corpus dictionary with each unique term in the corpus
        # and count the number of documents each term is exists in
        # e.g. global vocubulary
        for word in list(wordD.keys()):
            if word not in corpusD:
                corpusD[word] = CorpusTerm()
                
            corpusD[word].add_corpus_instance()

        # Store the doc:wordD association
        # my reprentation of a document (e.g., not a spacy document)
        docL.append( Doc(spacy_doc,filename,wordD) )
        

    # calculate TF/IDF for each word in each document (Doc()) and update corpusD
    for doc in docL:
        corpusD = doc.calc_tf_idf(len(docL),corpusD)

            
    # Form a sorted list of the average TF/IDF for each term in the corpus
    # output tuple: word, average tfidf, number of times term appeards in corpus
    tfidfL = [ (word, corpTerm.tfidf/corpusD[word].tfidfN,corpTerm.termN) for word,corpTerm in corpusD.items() ]
    tfidfL = [ ([word],tfidf,[termN]) for word,tfidf,termN in tfidfL if tfidf >= cfg.minTfIdfScore ]
    tfidfL = sorted( tfidfL, key=lambda x:x[1], reverse=True )

    # post processing on (e.g., Kenya and Kenyians)
    if cfg.mergeKeywordSubStringsFl:
        tfidfL = merge_sub_string(tfidfL,cfg.corpusKeyWordCount)
    
    # Generate the output
    # the reason why we looked at per document is because it was a way of making all documents have "n" entries
    # right way to do it is to say tf-ide per document, rather than over entire corpus
    if cfg.usePerDocWordCountFl:
        outL = generate_per_doc_output( cfg, docL, tfidfL )
    else:
        outL = generate_corpus_output( cfg, docL, tfidfL )

    return outL

def parse_yaml_cfg( fn ):
    """Parse the YAML configuration file."""

    if not os.path.isfile(fn):
        logging.error("The configuration file '%s' does not exists.",fn)
        return None
        
    cfg  = None

    with open(fn,"r") as f:
        cfgD = yaml.load(f, Loader=yaml.FullLoader)
        # create simplenamespace for clarity in referencing config inputs
        cfg = types.SimpleNamespace(**cfgD)
    return cfg
    

if __name__ == "__main__":

    def parse_args():
        """Parse the command line arguments."""

        ap = argparse.ArgumentParser(description="Find key words in a document corpus.")

        ap.add_argument("-c","--config_file",   default="keywords.yaml",     help="YAML configuration file.")
        ap.add_argument("-i","--input_dir",     default="test docs",         help="Corpus document directory.")
        ap.add_argument("-j","--json_out_file", default="keywords_out.json", help="JSON output file name.")
        ap.add_argument("-o","--html_out_file", default="index.html",        help="Optional HTML output file name.")
        ap.add_argument("-p","--print_summary", action='store_const', const=True, default=False, help="Print summary information.")

        return ap.parse_args()
    
    # parse the command line arguments
    args = parse_args()

    # set up logging - this is printing to STDOUT and STDERR
    logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO if args.print_summary else logging.ERROR )
    
    # parse the configuration file
    cfg = parse_yaml_cfg( args.config_file )

    if cfg:
        # run the program
        outL = main(cfg,args.input_dir)

        # write the output to a JSON file
        if outL:
            # write the JSON output
            with open(args.json_out_file,"w") as f:            
                f.write(json.dumps([ r.__dict__ for r in outL ]))

            # write the HTML output
            if len(args.html_out_file) > 0:
                write_html_output( outL, args.html_out_file )
