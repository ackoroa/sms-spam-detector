import numpy as np
import pandas as pd
from scipy import sparse
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

#Regex for web urls from https://github.com/rcompton/ryancompton.net/blob/master/assets/praw_drugs/urlmarker.py
WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

def derive_statistical_features(sms):
    sms['length'] = sms['message'].apply(len)
    sms['count_caps'] = sms['message'].apply(lambda msg: len(filter(lambda c: c in string.uppercase, msg)))
    sms['ratio_caps'] = sms['message'].apply(lambda msg: len(filter(lambda c: c in string.uppercase, msg)) / float(len(msg)))
    sms['count_digits'] = sms['message'].apply(lambda msg: len(filter(lambda c: c in string.digits, msg)))
    sms['ratio_digits'] = sms['message'].apply(lambda msg: len(filter(lambda c: c in string.digits, msg)) / float(len(msg)))
    sms['count_excl'] = sms['message'].apply(lambda msg: len(filter(lambda c: c == '!', msg)))

    # normalise count features to [0,1]
    sms['length'] = sms['length'].apply(lambda count: float(count) / sms['length'].max())
    sms['count_caps'] = sms['count_caps'].apply(lambda count: float(count) / sms['count_caps'].max())
    sms['count_digits'] = sms['count_digits'].apply(lambda count: float(count) / sms['count_digits'].max())
    sms['count_excl'] = sms['count_excl'].apply(lambda count: float(count) / sms['count_excl'].max())

def clean_message(sms):
    sms['message'] = sms['message'].apply(string.lower)
    sms['message'] = sms['message'].str.replace(WEB_URL_REGEX,' someurl ')
    sms['message'] = sms['message'].str.replace(r'[^\w\s]','')
    sms['message'] = sms['message'].str.replace(r'\d{5,}',' suspectnumber ')

def get_preprocessed_data(datafile):
    sms = pd.read_csv(datafile, encoding='latin-1')
    sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    sms = sms.rename(columns = {'v1':'label','v2':'message'})
    
    derive_statistical_features(sms)
    clean_message(sms)

    return sms

def get_classifiers():
    classifiers = []
    classifiers.append(LogisticRegression(solver='liblinear', penalty='l1'))
    classifiers.append(LogisticRegression(solver='liblinear', penalty='l2'))
    classifiers.append(SVC(kernel='linear'))
    classifiers.append(SVC(kernel='sigmoid', gamma=1.0))
    classifiers.append(MultinomialNB(alpha=0.2))
    classifiers.append(KNeighborsClassifier(n_neighbors=49))
    classifiers.append(DecisionTreeClassifier(min_samples_split=7))
    classifiers.append(RandomForestClassifier(n_estimators=31))
    classifiers.append(ExtraTreesClassifier(n_estimators=9))
    classifiers.append(AdaBoostClassifier(n_estimators=62))
    
    return classifiers

def train_and_score(features, labels, classifiers):
    # split the train and test data
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.7, random_state=0)

    # training and testing part --> can change to other classifiers
    scores_train = []
    scores = []
    for clf in classifiers:
        clf.fit(features_train, labels_train)
        labels_predicted_train = clf.predict(features_train)
        labels_predicted = clf.predict(features_test)
        scores_train.append(precision_recall_fscore_support(labels_train, labels_predicted_train))
        scores.append(precision_recall_fscore_support(labels_test, labels_predicted))
    return scores, scores_train

def str_precision_recall_fscore(scores, scores_train, classifier):
    s = classifier + ',ham,' + '%0.3f' % scores_train[0][0] + ',' + '%0.3f' % scores_train[1][0] + ',' + '%0.3f' % scores_train[2][0]
    s += ',' + '%0.3f' % scores[0][0] + ',' + '%0.3f' % scores[1][0] + ',' + '%0.3f' % scores[2][0] + '\n'
    s += ',spam,' + '%0.3f' % scores_train[0][1] + ',' + '%0.3f' % scores_train[1][1] + ',' + '%0.3f' % scores_train[2][1]
    s += ',' + '%0.3f' % scores[0][1] + ',' + '%0.3f' % scores[1][1] + ',' + '%0.3f' % scores[2][1]
    return s

def print_precision_recall_fscore(scores, scores_train, classifiers):
    print 'Classifier,Class,Training,,,Test,,'
    print ',,precision,recall,fscore,precision,recall,fscore'
    for i in range(len(scores)):
        print str_precision_recall_fscore(scores[i], scores_train[i], classifiers[i].__class__.__name__)
    print ''

if __name__ == "__main__":
    sms = get_preprocessed_data('spam.csv')

#    vectorizer = CountVectorizer(ngram_range=(1,1), decode_error='ignore', stop_words='english')
    vectorizer = TfidfVectorizer(ngram_range=(1,1), decode_error='ignore', stop_words='english')
    features_tfidf = vectorizer.fit_transform(sms['message'])
    features_others = sms.as_matrix(columns=['length','count_caps','ratio_caps','count_digits','ratio_digits','count_excl'])
    features = sparse.hstack((features_tfidf, features_others))

    classifiers = get_classifiers()

    print 'training with combined features'
    scores, scores_train = train_and_score(features, sms['label'], classifiers)
    print_precision_recall_fscore(scores, scores_train, classifiers)

    #print vectorizer.vocabulary_
    coef = classifiers[0].coef_[0]#.A[0]
    ham_coef_idx = np.argpartition(coef, 10)[10:]
    spam_coef_idx = np.argpartition(coef, -10)[-10:]

    print 'ham'
    for i in range(10):
        for name, idx in vectorizer.vocabulary_.iteritems():
            if idx == ham_coef_idx[i]:
                print i, name, coef[idx]
                break
    print ''
    print 'spam'
    for i in range(10):
        found = False
        for name, idx in vectorizer.vocabulary_.iteritems():
            if idx == spam_coef_idx[i]:
                found = True
                print i, name, coef[idx]
                break
        if not found:
            print i, '(not found)', spam_coef_idx[i], coef[spam_coef_idx[i]]
    print ''

    print 'tf-idf feature length:', features_tfidf.shape[1]
    print 'rest of features:', 'length,count_caps,ratio_caps,count_digits,ratio_digits,count_excl'
    for i in range(features_tfidf.shape[1], len(coef)):
        print i, coef[i]