import csv
import re, string, unicodedata, numpy as np
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_FILE_NAME = "wiki_movie_plots_deduped.csv"
CSV_FILE_SHORT_NAME = "wiki_movie_plots_deduped_short.csv"
PRE_PROC_FILE = "wiki_movie_plots_preprocessed"
TITLE_COL = 1
GENRE_COL = 5
PLOT_COL = 7
moviePlots = {}
headers = ""
DO_PREPROCESSING = False

LIMIT_GENRES = ['war', 'drama', 'comedy', 'thriller', 'action', 'romance', 'science fiction', 'crime', 'fantasy', 'musical', 'horror', 'mystery', 'adventure' ]

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

    def __init__(self, title, genre, rawPlot="", lemmas=None):
        self.title = title
        self.genre = genre
        self.rawPlot = rawPlot
        if lemmas == None:
            self.lemmas = set(lemmatize_verbs(normalize(nltk.word_tokenize(fix_contractions(rawPlot)))))
        else:
            self.lemmas = lemmas
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

if DO_PREPROCESSING:
    with open(CSV_FILE_NAME, encoding="utf8") as moviePlotDataset:
        csv_reader = csv.reader(moviePlotDataset, delimiter=',')
        firstLine = True
        for row in csv_reader:
            if firstLine:
                headers = row
                firstLine = not firstLine
            else:
                movies[row[TITLE_COL]] = extractMovieData(row)

    # save pre-processed to file
    with open(PRE_PROC_FILE, 'w', encoding="utf8") as moviePlotDatasetPreProcessed:
        for title in movies:
            movie = movies[title]
            moviePlotDatasetPreProcessed.write(movie.title + "\t" + movie.genre + '\t' + ' '.join(movie.lemmas))
            moviePlotDatasetPreProcessed.write(" \n")
        moviePlotDatasetPreProcessed.close()
        print("finished writing pre processed movies")
else:
    # read pre-processed to file
    with open(PRE_PROC_FILE, encoding="utf8") as file:
        for line in file:
            arr = line.split("\t")
            gList = arr[1].split(', / [ ] ;')
            gx = ''
            for g in gList:
                if g in LIMIT_GENRES:
                    gx = g
                    break
            if gx != '':
                movies[arr[0]] = Movie(title=arr[0], genre=gx, lemmas=set(arr[2].split(" ")))


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

""""""""""""""""""""""""""""""""""""
""" kNN method  """

def kNN(train_data, target_data):
    knn_naive_dv = KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine')
    knn_naive_dv = knn_naive_dv.fit(train_data, target_data)

    return knn_naive_dv;

lemmas = list(map(lambda x: ' '.join(x.lemmas), movies.values()))
genre = list(map(lambda x: x.genre, movies.values()))

unique_genres = set(genre)
data_count = len(lemmas)

vectorizer = TfidfVectorizer(min_df=2, tokenizer=None, preprocessor=None, stop_words=None, max_features=20000)
data_features = vectorizer.fit_transform(lemmas)
data_features = data_features.toarray()

[lemmas_train, lemmas_test, genre_train, genre_test] = model_selection.train_test_split(data_features, genre, test_size=0.3, random_state=66)

knn_model = kNN(lemmas_train, genre_train)

train_prediction = knn_model.predict(lemmas_train)
test_prediction = knn_model.predict(lemmas_test)

correct_pred_train = [i for i, j in zip(train_prediction, genre_train) if i == j]
correct_pred_test = [i for i, j in zip(test_prediction, genre_test) if i == j]

accuracy_train = len(correct_pred_train)/len(genre_train)
accuracy_test = len(correct_pred_test)/len(genre_test)

print("accuracy_train: " + str(accuracy_train))
print("accuracy_test: " + str(accuracy_test))
