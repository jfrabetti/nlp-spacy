# nlp-test
Produce a list of the most frequent interesting words, along with a summary table showing where those words appear (sentences and documents). 


# Installation

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm


# Usage

    A Python program that uses TF-IDF to determine the importance of a word in a corpus of similar documents. Output is provided in JSON and HTML format.

    Sample commands:
      python keywords.py 
      python keywords.py -o index.html  # Open index.html in browser to view output

```
    python keywords.py [-h] [-c CONFIG_FILE] [-i INPUT_DIR] [-j JSON_OUT_FILE] [-o HTML_OUT_FILE] [-p]

Arguments:

  -h, --help            				Show contencts of README and exit.
  
  -c CONFIG_FILE, --config_file CONFIG_FILE       	YAML configuration file.				Default: keywords.yaml
                        
			
  -i INPUT_DIR, --input_dir INPUT_DIR             	Corpus document directory.				Default: 'test doc'
                        
			
  -j JSON_OUT_FILE, --json_out_file JSON_OUT_FILE 	JSON output file name.					Default: keywords_out.json
                        
			
  -o HTML_OUT_FILE, --html_out_file HTML_OUT_FILE	HTML output file name.					Default: index.html
                        
			
  -p, --print_summary   				Print summary information.

```

# Configuration

The configuration file keywords.yaml may be used to adjust various input parameters in the keywords.py program. For example, by default the program considers a minimum of 25 corpus-wide important words to consider, but this may be changed uisng the 'corpusKeyWordCount' configuration.

Comments in the keywords.yaml provides information about the purpose of each input parameter.



