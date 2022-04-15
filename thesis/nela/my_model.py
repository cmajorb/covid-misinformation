import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

np.random.seed(500)

df = pd.read_csv("final.csv")
df = df[["input","output"]]
stop_words = set(stopwords.words('english'))

def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([str(ps.stem(word)) for word in text.split()])
    return text

df['input'].dropna(inplace=True)
df["input"] = df["input"].str.lower()

df['input'] = [re.sub(r'http\S+', '', entry) for entry in df['input']]


df["input"] = df['input'].str.replace('[^\w\s]','')
df['input'] = df["input"].apply(lambda x: ' '.join([str(word) for word in str(x).split() if word not in (stop_words)]))

df['input']=df['input'].apply(stemmer)

df = df.dropna()

print(df['output'].value_counts())

#df['input'] = df.apply(lambda row: nltk.word_tokenize(row['input']), axis=1)

#df['input'] = df["input"].apply(lambda x: ' '.join([str(word) for word in str(x).split() if word not in (stop_words)]))


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import classification_report

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['input'])
Encoder = LabelEncoder()
Input = Tfidf_vect.transform(df['input'])
Output = Encoder.fit_transform(df['output'])

n = 10
kf = KFold(n_splits=n,shuffle=True)
kf.get_n_splits(Input)

def trainModels(model):
    results = {'0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}}
    for train_index, test_index in kf.split(Input):
        Train_X, Test_X = Input[train_index], Input[test_index]
        Train_Y, Test_Y = Output[train_index], Output[test_index]
        #ros = RandomUnderSampler(random_state=0)
        #Train_X, Train_Y = ros.fit_resample(Train_X, Train_Y)
        model.fit(Train_X,Train_Y)
        Pred_Y = model.predict(Test_X)
        #print(classification_report(Test_Y, Pred_Y, digits=4))
        cr = classification_report(Test_Y, Pred_Y, digits=4,output_dict=True)
        for key in cr:
            if key != 'accuracy':
                for key2 in cr[key]:
                    results[key][key2].append(cr[key][key2])
            else:
                results[key].append(cr[key])
    for key in cr:
        if key != 'accuracy':
            for key2 in cr[key]:
                results[key][key2] = np.mean(results[key][key2])
        else:
            results[key] = np.mean(results[key])
    results.update({"accuracy": {"precision": None, "recall": None, "f1-score": results["accuracy"], "support": results['macro avg']['support']}})
    print(pd.DataFrame(results).transpose())
#Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Input, Output,test_size=0.25)

print(len(Train_Y))
print(sorted(Counter(Train_Y).items()))



print("Label stats:")
print(np.bincount(Train_Y))
print(np.bincount(Test_Y))

strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
trainModels(SVM)

'''
scores = cross_val_score(SVM, Total_X,Total_Y, cv=strat_k_fold)
print("SVM")
print(np.std(scores))
print(np.mean(scores))
'''
from sklearn.utils.random import sample_without_replacement



from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
trainModels(lr)

'''
scores = cross_val_score(lr, Total_X,Total_Y, cv=strat_k_fold)
print("LR")
print(np.std(scores))
print(np.mean(scores))
'''


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier() 
trainModels(forest)


'''
scores = cross_val_score(forest, Total_X,Total_Y, cv=strat_k_fold)
print("RF")
print(np.std(scores))
print(np.mean(scores))
'''

'''
cv=CountVectorizer(binary=True, ngram_range=(3,3), lowercase=False)
#transformed train reviews

trigram_feature_cv_train = cv.fit_transform(Train_X).toarray()
trigram_feature_cv_test = cv.transform(Test_X).toarray()

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(trigram_feature_cv_train,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(trigram_feature_cv_test)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

print(roc_auc_score(predictions_SVM, Test_Y)*100)
'''