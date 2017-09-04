from collections import defaultdict
from collections import Counter
import collections
import unicodedata
from unidecode import unidecode
from textblob import TextBlob
#from helpers import twitter_tokenizer
import numpy as np
from helpers import constants
import nltk
import re
from nltk.tokenize.treebank import TreebankWordTokenizer
from textblob.taggers import NLTKTagger

#proc = CoreNLP(configdict={'annotators':'tokenize, ssplit'}, corenlp_jars=[constants.STANFORD_CORENLP_PATH])
#pos_proc = CoreNLP(configdict={'annotators':'tokenize, ssplit, pos, lemma, ner'}, corenlp_jars=[constants.STANFORD_CORENLP_PATH])

treebank_tokenizer = TreebankWordTokenizer()
nltk_tagger = NLTKTagger()
caps = "([A-Z])"
prefixes = "(\$|Ecl|Col|no|Rs|S|Fr|Op|J|Bros|al|vs|HMA|Card|Corp|No|c|v|Mr|St|Mrs|Ms|Dr)[.]"
months = "(Sept|Oct|Nov|Dec|Jan|Feb|Mar|Apr|Jun|Jul|Aug)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|png|jpeg|ipg|zip|tj|ru|mp3|mp4|[0-9]|ts|m2ts)"
digits = "([0-9])"

stopwords_list = ["a", "about", "above", "above", "does", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
#stopwords_list = ["where", "what", "name", "a", "is", "of", "who", "why", "when", "was", "which", "what's"]
verb_list = ["the", "does"]

# Append location to path
nltk.data.path.append(constants.NLTK_DATA_PATH)

def tag_sentence(sentence):
	"""
	Part-of-speech tags a sentence
	Returns words_list, tags_list
	"""
	blob = TextBlob(sentence, tokenizer=treebank_tokenizer, pos_tagger=nltk_tagger)
	words = blob.tags
	words_list = map(lambda word: word[0], words)
	tags_list = map(lambda word: word[1], words)
	return words_list, tags_list

def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

def ner_sentence(sentence):
	sentences = pos_proc.parse_doc(sentence)
	words = []
	pos = []
	ner = []
	for sentence in sentences['sentences']:
		words.extend(sentence['tokens'])
		pos.extend(sentence['pos'])
		ner.extend(sentence['ner'])
	return words, pos, ner

class WhitespaceTokenizer(object):
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

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
    print(toks[0:10])
    for m in re.finditer(pattern, tags):
        start_index = tags[0:m.start()].count(' ')
        end_index = tags[0:m.end()].count(' ')
        #yield (start_index, end_index)
        yield start_index, end_index, doc[tags[0:m.start()].count(' '):tags[0:m.end()].count(' ')]

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
    ent_mappings = defaultdict(list)

    for ent in doc.ents:
        if "CNN" not in ent.text:
            ent_mappings[ent.label_].append(ent)

    filtered_verb_phrases = filter(lambda vp: len(vp) > 1, verb_phrases)
    filtered_noun_phrases = filter(lambda np: len(np) > 1, noun_phrases)
    
    for k in ent_mappings:
        entities = ent_mappings[k]

        print(entities)
        print(entities[0])
        if len(entities) == 1: 
            random_entities = [entities]
        else:
            random_entities = sample(entities, num_samples)

        for i in range(0, len(random_entities)):
            ent = entities[i]
            
            print("On index %s" % i)
            start_indices.append(ent.start)
            end_indices.append(ent.end)

    random_noun_phrases = sample(noun_phrases, num_samples)
    random_verb_phrases = sample(verb_phrases, num_samples)

    for phrase in random_noun_phrases:
        start_indices.append(phrase[0])
        end_indices.append(phrase[1])

    for phrase in random_verb_phrases:
        start_indices.append(phrase[0])
        end_indices.append(phrase[1])


def tag_tokenize_sentence(sentence, vocab, pos_vocab, add_start_end=True):
	"""
	Tags and tokenizes a sentence
	"""
	words_list, tags_list = tag_sentence(sentence)
	tag_tokens = pos_vocab.map(tags_list, add_start_end)
	if vocab.vocab_type == constants.WORD_CHAR_LEVEL:
		word_tokens = vocab.map_list(words_list, add_start_end)
	elif vocab.vocab_type == constants.WORD_LEVEL:
		word_tokens = vocab.map(words_list, add_start_end)
	else:
		raise Exception("Invalid vocab type (only works on word level) %s" % vocab.vocab_type)
	return word_tokens, tag_tokens

def clean_spaces(text):
	"""
	Returns text with special space characters replaced with " "
	text: text to replace
	"""
	cleaned_text = ' '.join(text.split())
	return cleaned_text

def tokenize_paragraph(paragraph, vocab, tokenizer_type=constants.TOKENIZER_NLTK, add_start_end=True):
	"""
	Splits and tokenizes a paragraph
	"""
	sentences = split_paragraph(paragraph)
	tokens = map(lambda sentence: tokenize_sentence(sentence, vocab, \
		tokenizer_type=tokenizer_type, \
		add_start_end=add_start_end), sentences)
	return tokens

def get_ngrams(array, n):
	ngrams = [array[i:i+n] for i in range(0, len(array) - n + 1)]
	return ngrams 

def get_word_trigrams(word):
	"""
	Converts word into its trigram level representation
	i.e. cat -> #cat# -> [#ca, cat, at#]
	"""
	padded_word = "%s%s%s" % (constants.WORD_HASHING_CONSTANT, word, constants.WORD_HASHING_CONSTANT)
	word_length = len(padded_word)
	trigrams = [padded_word[i:i+3] for i in xrange(word_length - 2)]
	return trigrams

def remove_non_ascii(str):
	res = "".join([x if ord(x) < 128 else ' ' for x in str])
	return res

def tokenize_sentence(sentence, vocab, tokenizer_type=constants.TOKENIZER_NLTK, add_start_end=True):
	"""
	Splits and tokenizes a sentence
	"""
	if vocab.vocab_type == constants.WORD_CHAR_LEVEL:
		words = split_sentence(sentence, tokenizer_type=tokenizer_type)
		tokens = vocab.map_list(words, add_start_end)
	elif vocab.vocab_type == constants.WORD_HASHING_LEVEL:
		words = split_sentence(sentence, tokenizer_type=tokenizer_type)
		word_trigrams = map(lambda word: get_word_trigrams(word), words)
		tokens = vocab.map_list(word_trigrams, add_start_end)
	elif vocab.vocab_type == constants.WORD_LEVEL:
		words = split_sentence(sentence, tokenizer_type=tokenizer_type)
		tokens = vocab.map(words, add_start_end)
	elif vocab.vocab_type == constants.CHAR_LEVEL:
		tokens = vocab.map(sentence, add_start_end)
	else:
		raise Exception("Invalid vocab type %s" % vocab.vocab_type)
	return tokens

def split_paragraph(text, tokenizer_type=constants.TOKENIZER_REGEX):
	sentences = ""

	if tokenizer_type == constants.TOKENIZER_NLTK:
		sentences = nltk.sent_tokenize(paragraph)
	elif tokenizer_type == constants.TOKENIZER_REGEX:
		if '.' in text:
			text = " " + text + "  "
			text = text.replace("\n"," ")
			text = re.sub("Robots[.]txt", "Robots<prd>txt", text)
			text = re.sub("robots[.]txt", "robots<prd>txt", text)
			text = re.sub("p[.]m", "p<prd>m", text)
			text = re.sub("a[.]m[.]", "a<prd>m<prd>", text)
			text = re.sub("A[.]M[.]", "A<prd>M<prd>", text)
			text = re.sub("p[.]m[.]", "p<prd>m<prd>", text)
			text = re.sub("P[.]m[.]", "P<prd>m<prd>", text)
			text = re.sub("P[.]M[.]", "P<prd>M<prd>", text)
			text = re.sub("S[.]B[.]", "S<prd>B<prd>", text)
			text = re.sub(months, "\\1<prd>", text)
			text = re.sub("J[.]S[.]", "J<prd>S<prd>", text)
			text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
			text = re.sub(prefixes,"\\1<prd>",text)
			text = re.sub("H[.]R[.]", "H<prd>R<prd>",text)
			text = re.sub("h[.]r[.]", "h<prd>r<prd>",text)
			text = re.sub("i[.]e[.]", "i<prd>e<prd>",text)
			text = re.sub("e[.]g[.]", "e<prd>g<prd>",text)
			text = re.sub("e[.]g", "e<prd>g",text)
			text = re.sub("U[.]S[.]", "U<prd>S<prd>", text)
			text = re.sub("Lt.", "Lt<prd>", text)
			text = re.sub("[.][.][.]", "<prd><prd><prd>", text)
			text = re.sub(websites,"<prd>\\1",text)
			if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
			text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
			text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
			text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
			text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
			text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
			text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
			text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
			if "\"" in text: text = text.replace(".\"","<prd>\"<stop>")
			if "!" in text: text = text.replace("!\"","!\"<stop>")
			if "?" in text: text = text.replace("?\"","?\"<stop>")
			text = text.replace(".",".<stop>")
			text = text.replace("?","?<stop>")
			text = text.replace("!","!<stop>")
			text = text.replace("<prd>",".")
			sentences = text.split("<stop>")

			if sentences[-1] == "  ":
				sentences = sentences[:-1]

			sentences = [s.strip() for s in sentences]
			sentences = filter(lambda sentence: len(sentence) > 0, sentences)
		else:
			sentences = [text]
	elif tokenizer_type == constants.TOKENIZER_TEXTBLOB:
		textblob = TextBlob(text)
		sentences = map(lambda sentence: sentence.raw, textblob.sentences)
	elif tokenizer_type == constants.TOKENIZER_TWITTER:
		print("No twitter tokenizer implemented. Using textblob")
		textblob = TextBlob(text)
		sentences = map(lambda sentence: sentence.raw, textblob.sentences)
	else:
		raise Exception("Invalid tokenizer type given %s" % tokenizer_type)

	return sentences

def space_out_punct(sentence, punct_list=["--", "-", "/", "'", "\""]):
	new_sentence = sentence
	for punct in punct_list:
		new_sentence = new_sentence.replace(punct, " " + punct + " ")
	return new_sentence

def split_sentence(sentence, tokenizer_type=constants.TOKENIZER_TEXTBLOB):
    """
    Splits sentence into a list of words
    """
    if tokenizer_type == constants.TOKENIZER_TEXTBLOB:
        textblob = TextBlob(sentence, tokenizer=treebank_tokenizer, pos_tagger=nltk_tagger)
        words = textblob.words
    elif tokenizer_type == constants.TOKENIZER_NLTK:
        sent = space_out_punct(sentence)
        words = nltk.word_tokenize(sent)
    elif tokenizer_type == constants.TOKENIZER_REGEX:
        words = sentence.split()
    elif tokenizer_type == constants.TOKENIZER_TWITTER:
        sent = space_out_punct(sentence)
        words = twitter_tokenizer.tokenize(sent)
    elif tokenizer_type == constants.TOKENIZER_SPECIAL_DELIMITER:
        words = sentence.split('*@#$*($#@*@#$')
    elif tokenizer_type == constants.TOKENIZER_STANFORD_NLP:
        sent = space_out_punct(sentence)
        sentences = proc.parse_doc(sent)
        words = []
        for sentence in sentences['sentences']:
        	for token in sentence['tokens']:
        		new_token = token
        		if "-RRB-" in token:
        			new_token = token.replace("-RRB-", "")
        			words.append(new_token)
        			words.append(")")
        		elif "-LRB" in token:
        			new_token = token.replace("-LRB-", "")
        			words.append(new_token)
        			words.append("(")
        		elif "-RSB-" in token:
        			new_token = token.replace("-RSB-", "]")
        			words.append(new_token)
        		elif "-LSB-" in token:
        			new_token = token.replace("-LSB-", "[")
        			words.append(new_token)
        		else:
        			words.append(new_token)
    elif tokenizer_type == constants.TOKENIZER_TAB:
        return sentence.split("\t")
    elif tokenizer_type == constants.TOKENIZER_SPACE:
        return sentence.split(" ")
    else:
        raise Exception("Invalid tokenizer given %s" % tokenizer_type)
    return words

def remove_stopwords(sentence):
	sentence = sentence.replace("?", "")
	sentence = sentence.replace("'", ' ')
	items = sentence.split(' ')
	cleaned_string = map(lambda item:clean_token(item, stopwords_list), items)
	string_w_spaces = ' '.join(cleaned_string)
	raw_tokens = string_w_spaces.split()
	single_space_string = ' '.join(raw_tokens)

	if len(raw_tokens) < 1:
		return sentence
	else:
		return single_space_string

def clean_token(token, stopwords_list):
	if token.lower() in stopwords_list or token == "s" or token in verb_list:
		return ""
	else:
		return token

def remove_non_ascii_characters(string):
	cleaned_string = (lambda s: "".join(i for i in s if 31 < ord(i) < 127), string)
	return cleaned_string

def replace_accents(token):
	""" Replaces accents with special value """
	utf8_str = token.decode('utf-8')
	normalized_str = utf8_str.encode('utf-8')
	normalized_str = normalized_str.replace("'", " ' ")
	return normalized_str

def clean_name_arr(name_arr, query):
	""" Only returns values from name_dict whose keys are a substring of query 
		name_dict: maps names to ids, keys
	"""
	correct_names = []

	query = query + " "
	lowercase_query = query.lower()
	quote_removed_query = lowercase_query.replace('\\"', '')
	question_removed_query = lowercase_query.replace('?', '')
	quote_removed_question_query = lowercase_query.replace('"', '').replace('?', '')

	for k in name_arr:
		spaced_k = k.lower() + " "
		if spaced_k in lowercase_query or \
		spaced_k in quote_removed_query or \
		spaced_k in question_removed_query or \
		spaced_k in quote_removed_question_query:
			correct_names.append(k)

	return correct_names

def remove_substrings_arr(substring_arr):
	""" Remove any string in array that is a substring in another string 
	"""
	substring_set = set(substring_arr)
	filtered_items = filter(lambda item: not is_substring(item, substring_set), substring_arr)
	return filtered_items

def is_substring(string, string_set):
	"""
	Returns true if string is a substring of any string in 
	string_set that is not equal to string
	"""
	substrings = filter(lambda cur_string: (string in cur_string) and string != cur_string, string_set)
	is_substring = len(substrings) > 0
	return is_substring

def clean_name_dict(name_dict, query):
	""" Only returns values from name_dict whose keys are a substring of query 
		name_dict: maps names to ids, keys
	"""
	correct_names = dict()

	lowercase_query = query.lower()
	for k, v in name_dict.iteritems():
		if k.lower() in lowercase_query:
			correct_names[k] = v

	return correct_names

def replace_tokens(token, max_length):
	if len(token) > max_length:
		print("Token bigger than max length " + str(len(token)))
		new_token = token[0:max_length]
	else:
		new_token = token
	return new_token

def get_tokens(text, vocab, delimiter, max_length, use_tokenizer=False):
	tokens = replace_tokens(vocab.string_to_tokens(text=text, \
				delimiter=delimiter, \
				add_start_end=True, use_tokenizer=use_tokenizer), max_length)
	return tokens

# Get tokens for text array
def get_tokens_arr(text_arr, vocab, delimiter, max_length, use_tokenizer=False):
	arr_tokens = map(lambda text:\
		get_tokens(text, vocab, delimiter, max_length, use_tokenizer=use_tokenizer), \
		text_arr)
	sizes = map(lambda token:len(token), arr_tokens)

	if vocab.vocab_type == constants.WORD_CHAR_LEVEL:
		word_sizes = map(lambda tokens:np.max(\
			map(lambda token:len(token), tokens)), arr_tokens)
		max_word_size = np.max(word_sizes)
	else:
		max_word_size = 0
	max_size = np.max(sizes)
	return arr_tokens, max_size, max_word_size

def hashable(obj):
    if isinstance(obj, collections.Hashable):
        items = obj
    elif isinstance(obj, collections.Mapping):
        items = frozenset((k, hashable(v)) for k, v in obj.iteritems())
    elif isinstance(obj, collections.Iterable):
        items = tuple(hashable(item) for item in obj)
    else:
        raise TypeError(type(obj))

    return items

def valid_intersection(substring, gold_substring, max_intersection):
	first_set = set(hashable(substring))
	second_set = set(hashable(gold_substring))
	intersection = first_set.intersection(second_set)
	num_elements = len(intersection)

	is_valid = num_elements <= max_intersection
	is_subelement = substring in gold_substring and len(substring) == max_intersection
	return is_valid and not is_subelement

def get_all_substrings(input_string, max_length=None, exact_length=None):
	"""
	Returns all substrings of a string
	max_length: if specified returns all strings of at most this length
	exact_length: if specified returns all strings of this exact length
	"""
	length = len(input_string)
	substrings = [input_string[i:j+1] for i in xrange(length) for j in xrange(i,length)]

	if max_length != None:
		filtered_substrings = filter(lambda substr:len(substr) <= max_length, substrings)
		return filtered_substrings
	elif exact_length != None:
		filtered_substrings = filter(lambda substr:len(substr) == exact_length, substrings)
		return filtered_substrings
	else:
		return substrings

def get_all_valid_substrings(substrings, gold_substring, max_intersection):
	valid_substrings = filter(lambda substring: valid_intersection(substring, gold_substring, max_intersection), substrings)
	return valid_substrings

def add_tokens_to_arr_word(tokens_arr, \
	sequence_lengths_arr, \
	tokens, \
	batch_index, \
	max_length):
	"""
	Add tokens to tokens array at specified batch_index. 
	tokens_arr: Array to add tokens to 
	sequence_lengths_arr: Array to add sequence lengths to 
	word_lengths_arr: Array to add word lengths to 
	tokens: tokens to add to tokens_arr
	batch_index: index to batch to
	max_length: max length of sequence length 
	max_word_length: max length of word tokens
	"""

	tokens_length = int(np.min([len(tokens), max_length]))

	for j in range(0, tokens_length):
		cur_word = tokens[j]
		tokens_arr[batch_index][j] = cur_word

	sequence_lengths_arr[batch_index] = tokens_length

def add_tokens_to_arr(tokens_arr, \
	sequence_lengths_arr, \
	word_lengths_arr, \
	tokens, \
	batch_index, \
	max_length, \
	max_word_length):
	"""
	Add tokens to tokens array at specified batch_index. 
	tokens_arr: Array to add tokens to 
	sequence_lengths_arr: Array to add sequence lengths to 
	word_lengths_arr: Array to add word lengths to 
	tokens: tokens to add to tokens_arr
	batch_index: index to batch to
	max_length: max length of sequence length 
	max_word_length: max length of word tokens
	"""

	tokens_length = int(np.min([len(tokens), max_length]))

	for j in range(0, tokens_length):
		cur_word = tokens[j]
		cur_word_length = int(np.min([len(cur_word), max_word_length]))
		tokens_arr[batch_index][j][0:cur_word_length] = cur_word[0:cur_word_length]
		word_lengths_arr[batch_index][j] = cur_word_length

	sequence_lengths_arr[batch_index] = tokens_length

def add_tokens_to_arr_samples(tokens_arr, \
	sequence_lengths_arr, \
	word_lengths_arr, \
	tokens, \
	batch_index, \
	sample_index, \
	max_length, \
	max_word_length):

	"""
	Add tokens to tokens array at specified batch_index. 
	tokens_arr: Array to add tokens to 
	sequence_lengths_arr: Array to add sequence lengths to 
	word_lengths_arr: Array to add word lengths to 
	tokens: tokens to add to tokens_arr
	batch_index: index to batch to
	sample_index: index to add sample to
	max_length: max length of sequence length 
	max_word_length: max length of word tokens
	"""
	tokens_length = np.min([len(tokens), max_length])

	for j in range(0, tokens_length):
		cur_word = tokens[j]
		cur_word_length = np.min([len(cur_word), max_word_length])
		tokens_arr[batch_index][sample_index][j][0:cur_word_length] = cur_word[0:cur_word_length]
		word_lengths_arr[batch_index][sample_index][j] = cur_word_length

	sequence_lengths_arr[batch_index][sample_index] = tokens_length