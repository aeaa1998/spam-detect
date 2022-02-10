import unicodedata
from string import punctuation
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, accuracy_score

def removerAcentos(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ascii','ignore').decode('utf-8','ignore')
    return texto

def remove_trash(texto, removerDigitos =False):
    patron = r'[^a-zA-Z0-9\s]' if not removerDigitos else r'[^a-zA-Z\s]'
    texto = re.sub(patron,'', texto)
    return texto

# Eliminate punctuations and accents
def remove_msg_punctuations(email_msg):
    puntuation_removed_msg = "".join([word for word in unicodedata.normalize('NFD', email_msg) if word not in punctuation and unicodedata.category(word) != 'Mn'])
    return puntuation_removed_msg


# We will tokenize
# \W+ Word character regex [A-Za-z0-9_] at least one time
tokenizer = nltk.RegexpTokenizer(r"\w+")
def tokenize_into_words(text):
    
    return tokenizer.tokenize(text)

# To remove common woeds
def remove_stop_words(tokenized_words):
    stop_words_en = stopwords.words("english")
    stop_words_spa = stopwords.words("spanish")
    return [token for token in tokenized_words if token not in stop_words_en and token not in stop_words_spa]

#lemmatizing
# To make the words in the most "basic case" of its meaning
word_lemmatizer = WordNetLemmatizer()
def lemmatization(tokenized_words):
    lemmatized_text = [word_lemmatizer.lemmatize(word)for word in tokenized_words]
    return lemmatized_text

# We will clean our message
def preprocess(message):
    # Remove the punctuations and lower them
    message = removerAcentos(remove_trash(message, True))
    message = remove_msg_punctuations(message).lower()
    # Make tokens with the body
    tokens = tokenize_into_words(message)
    # Lemmatize
    tokens = lemmatization(tokens)
#     Remove stop words
    clean = remove_stop_words(tokens)
    return ' '.join(clean)


def model_results(y_test, y_pred):
    """ Prints model results. """

    validation_confusion_matrix = confusion_matrix(y_test, y_pred)
    validation_recall_score = recall_score(y_test, y_pred, average=None)
    validation_precision_score = precision_score(y_test, y_pred, average=None)
    validation_f1_score = f1_score(y_test, y_pred, average=None)
    print("Matrix de confusión: \n", validation_confusion_matrix)
    print(classification_report(y_test, y_pred, target_names = ["legit", "dga"]))
    print("recall_score: ",validation_recall_score)
    print("precision_score: ", validation_precision_score)
    print("f1_score: ", validation_f1_score)
    print("accuracy: ", accuracy_score(y_test, y_pred))