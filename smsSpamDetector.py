import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    sms = pd.read_csv("spam.csv", encoding='latin-1')
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

    #print sms.head(20)

    vectorizer = CountVectorizer(ngram_range=(1,3), decode_error='ignore', stop_words="english")
    
    