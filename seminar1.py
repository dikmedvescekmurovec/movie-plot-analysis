import csv
import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


CSV_FILE_NAME = "wiki_movie_plots_deduped.csv"
TITLE_COL = 1
GENRE_COL = 5
PLOT_COL = 7
moviePlots = {}
headers = ""

""""""""""""""""""""""""""""""""""""
""" Text preprocessing functions """
""""""""""""""""""""""""""""""""""""

def fix_contractions(text):
    """Fix all contractions in text. Eg don't -> do not"""
    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

""""""""""""""""""""
""" Movie class """
""""""""""""""""""""
class Movie:

    def __init__(self, title, genre, rawPlot):
        self.title = title
        self.genre = genre
        self.rawPlot = rawPlot
        self.lemmas = set(lemmatize_verbs(normalize(nltk.word_tokenize(fix_contractions(rawPlot)))))
        self.lemmaCount = {}
        
    def countLemmas():
        for lemma in self.lemmas:
            if lemma not in lemmaCount:
                lemmaCount[lemma] = 1
            else:
                lemmaCount[lemma] += 1

    """ TODO: Store and read from CSV for later reuse. """
    def storeCSV():
        return False
    def readCSV():
        return False
def extractMovieData(csvRow):
    return Movie(csvRow[TITLE_COL], csvRow[GENRE_COL], csvRow[PLOT_COL])


movies = {}

with open(CSV_FILE_NAME, encoding="utf8") as moviePlotDataset:
    csv_reader = csv.reader(moviePlotDataset, delimiter=',')
    firstLine = True
    for row in csv_reader:
        print(row[TITLE_COL])
        if firstLine:
            headers = row
            firstLine = not firstLine
        else:
            movies[row[TITLE_COL]] = extractMovieData(row)
            
            
""""""""""""""""""""""""""""""""""""
""" Nearest neighbours method  """
""""""""""""""""""""""""""""""""""""
def nearestNeighbourGenre(movie):
    closestMovie = None
    closestDistance = 1
    for m in movies.values():
        if (m.title != movie.title):
            distance = nltk.jaccard_distance(movie.lemmas, m.lemmas)
            if distance < closestDistance:
                closestDistance = distance
                closestMovie = m
    print(closestMovie.title)
    return closestMovie.genre


print(nearestNeighbourGenre(movies["A Christmas Carol"]))

