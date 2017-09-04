import spacy
from spacy.tokens import Doc
import re 
import numpy as np 
from collections import defaultdict 


class WhitespaceTokenizer(object):
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en')
nlp.make_doc = WhitespaceTokenizer(nlp)

def pos_regex_matches(doc, pattern):
    """
    Extract sequences of consecutive tokens from a spacy-parsed doc whose
    part-of-speech tags match the specified regex pattern.

    Args:
        doc (``textacy.Doc`` or ``spacy.Doc`` or ``spacy.Span``)
        pattern (str): Pattern of consecutive POS tags whose corresponding words
            are to be extracted, inspired by the regex patterns used in NLTK's
            `nltk.chunk.regexp`. Tags are uppercase, from the universal tag set;
            delimited by < and >, which are basically converted to parentheses
            with spaces as needed to correctly extract matching word sequences;
            white space in the input doesn't matter.

            Examples (see ``constants.POS_REGEX_PATTERNS``):

            * noun phrase: r'<DET>? (<NOUN>+ <ADP|CONJ>)* <NOUN>+'
            * compound nouns: r'<NOUN>+'
            * verb phrase: r'<VERB>?<ADV>*<VERB>+'
            * prepositional phrase: r'<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+'

    Yields:
        ``spacy.Span``: the next span of consecutive tokens from ``doc`` whose
            parts-of-speech match ``pattern``, in order of apperance
    """
    # standardize and transform the regular expression pattern...
    pattern = re.sub(r'\s', '', pattern)
    pattern = re.sub(r'<([A-Z]+)\|([A-Z]+)>', r'( (\1|\2))', pattern)
    pattern = re.sub(r'<([A-Z]+)\|([A-Z]+)\|([A-Z]+)>', r'( (\1|\2|\3))', pattern)
    pattern = re.sub(r'<([A-Z]+)\|([A-Z]+)\|([A-Z]+)\|([A-Z]+)>', r'( (\1|\2|\3|\4))', pattern)
    pattern = re.sub(r'<([A-Z]+)>', r'( \1)', pattern)


    tags = ' ' + ' '.join(tok.pos_ for tok in doc)
    toks = list(map(lambda t: t.pos_, doc))

    for m in re.finditer(pattern, tags):
        start_index = tags[0:m.start()].count(' ')
        end_index = tags[0:m.end()].count(' ')
        #yield (start_index, end_index)
        yield start_index, end_index, doc[tags[0:m.start()].count(' '):tags[0:m.end()].count(' ')]

def extract_NER(text):
    doc = nlp(text)
    for ent in doc.ents:
        if "CNN" not in ent.text:
            ent_mappings[ent.label_].append(ent)

def sample(arr, num_samples):
    num_samples = np.min([len(arr), num_samples])
    entities = np.random.permutation(arr)[0:num_samples]
    return entities

def extract_phrases(text, num_samples=2):
    start_indices = []
    end_indices = []
    doc = nlp(text)

    noun_phrase_pattern = '<DET>? (<NOUN|PROPN>+ <ADP|CONJ|CCONJ|PUNCT>*)* <NOUN|PROPN>+'
    verb_phrase_pattern = '<VERB>?<ADV>*<VERB>+'
    prepositional_phrase_pattern = '<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+'

    noun_phrases = list(pos_regex_matches(doc, noun_phrase_pattern))
    verb_phrases = list(pos_regex_matches(doc, verb_phrase_pattern))
    noun_chunks = np.array(list(doc.noun_chunks))

    if len(noun_chunks) > 1:
        lengths = list(map(lambda l: len(l), noun_chunks))
        sorted_indices = np.argsort(lengths)
        top_sorted_indices = np.min([len(sorted_indices), 3])
        top_sorted_indices = sorted_indices[-top_sorted_indices:]
        top_chunks = noun_chunks[top_sorted_indices]

        for i in range(0, len(top_chunks)):
            cur_chunk = top_chunks[i]
            if type(cur_chunk) == type(np.array([])):
                print("Invalid chunk given")
                continue
            cur_start = cur_chunk.start 
            cur_end = cur_chunk.end 

            start_indices.append(cur_start)
            end_indices.append(cur_end)

            #print(cur_chunk)

    ent_mappings = defaultdict(list)

    for ent in doc.ents:
        if "CNN" not in ent.text:
            ent_mappings[ent.label_].append(ent)

    #print("Printing entities")
    #print(ent_mappings)

    filtered_verb_phrases = list(filter(lambda vp: len(vp[2]) > 1, verb_phrases))
    filtered_noun_phrases = list(filter(lambda np: len(np[2]) > 1, noun_phrases))

    for k in ent_mappings:
        entities = ent_mappings[k]

        if len(entities) == 1: 
            random_entities = [entities]
        else:
            random_entities = sample(entities, num_samples)

        for i in range(0, len(random_entities)):
            ent = entities[i]
            start_indices.append(ent.start)
            end_indices.append(ent.end)

    random_noun_phrases = sample(noun_phrases, num_samples)
    random_verb_phrases = sample(verb_phrases, num_samples)

    for phrase in random_noun_phrases:
        start_indices.append(phrase[0])
        end_indices.append(phrase[1])
        #print(doc[phrase[0]:phrase[1]])

    for phrase in random_verb_phrases:
        start_indices.append(phrase[0])
        end_indices.append(phrase[1])

    return start_indices, end_indices
