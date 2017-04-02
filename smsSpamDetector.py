import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from scipy import sparse

# NOTE TO ARNOLD: Need to download nltk package for stopwords
# pip install nltk
# open python
# import nltk
# nltk.download("stopwords")
# source : http://stackoverflow.com/questions/26693736/nltk-and-stopwords-fail-lookuperror

# this method removes stop words, puntuations.
# it returns the "cleaned" string
def process_text(text):
    text = text.translate(None, string.punctuation)
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

def train_and_score(features,labels):    
    # split the train and test data
    features_train, features_test,labels_train, labels_test = train_test_split(features, labels, train_size=0.7, random_state=0)

    # training and testing part --> can change to other classifiers
    clf = SVC(kernel='sigmoid', gamma=1.0)
    clf.fit(features_train,labels_train)
    return clf.score(features_test,labels_test)


if __name__ == "__main__":
    sms = pd.read_csv("spam.csv")
    sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    sms = sms.rename(columns = {'v1':'label','v2':'message'})
    
    sms['length'] = sms['message'].apply(len)
    sms['count_caps'] = sms['message'].apply(lambda msg: len(filter(lambda c: c in string.uppercase, msg)))
    sms['ratio_caps'] = sms['message'].apply(lambda msg: len(filter(lambda c: c in string.uppercase, msg)) / float(len(msg)))
    sms['count_digits'] = sms['message'].apply(lambda msg: len(filter(lambda c: c in string.digits, msg)))
    sms['ratio_digits'] = sms['message'].apply(lambda msg: len(filter(lambda c: c in string.digits, msg)) / float(len(msg)))
    sms['count_excl'] = sms['message'].apply(lambda msg: len(filter(lambda c: c == '!', msg)))
    
    #print sms.head()
    #print '\nham'
    #print sms[sms['label'] == 'ham'].describe()
    #print '\nspam'
    #print sms[sms['label'] == 'spam'].describe()

    sms['message'] = sms['message'].str.lower()
    sms['message'] = sms['message'].str.replace(r'www.\S+',' someurl ')
    sms['message'] = sms['message'].str.replace(r'\d{4,}',' suspectnumber ')
    sms['message'] = sms['message'].str.replace(r'[^\w\s\d]','')

    # print sms.head(20)

    vectorizer = CountVectorizer(ngram_range=(1,3), decode_error='ignore', stop_words="english")
    
    # tf.idf vectorizer
    messages = sms['message'].copy()
    messages = messages.apply(process_text)
    vectorizer = TfidfVectorizer("english")
    features_tfidf = vectorizer.fit_transform(messages)

    # construct other features
    features_others = sms.loc[:,['length','count_caps','ratio_caps','count_digits','ratio_digits','count_excl']]

    # combine the two features
    features = sparse.hstack((features_tfidf,features_others)).A

    print 'training with tfidf features'
    print 'accuracy = ' + str(train_and_score(features_tfidf,sms['label']))

    print 'training with other features'
    print 'accuracy = '  + str(train_and_score(features_others,sms['label']))

    print 'training with combined features'
    print 'accuracy = ' + str(train_and_score(features,sms['label']))
