import datetime
import sys
import time
import json
import pickle
import numpy as np
import io

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

from data_helper import data_folder, file_list, result_folder

try:
    from_id = int(sys.argv[1])
    to_id = int(sys.argv[2])
except:
    from_id = 0
    to_id = len(file_list)

n_classifiers = int(sys.argv[3])

meta_result_kne = []
meta_result_mdes = []
meta_result_knu = []


def get_xgboost_classifier(n_classes):
    if n_classes == 2:
        return XGBClassifier(n_estimators=200, object='binary:logistic')
    else:
        return XGBClassifier(n_estimators=200, object='multi:softmax')

for i_file in range(from_id, to_id):
    try:
        file_name = file_list[i_file]
        print(datetime.datetime.now(), ' File {}: '.format(i_file), file_name)

    #-------------------DATA PREPROCESS---------------------
        D_train = np.loadtxt(data_folder + '/train1/' + file_name + '_train1.dat', delimiter=',')
        D_val = np.loadtxt(data_folder + '/val/' + file_name + '_val.dat', delimiter=',')
        D_test = np.loadtxt(data_folder + '/test/' + file_name + '_test.dat', delimiter=',')

        X_train = D_train[:, :-1]
        Y_train = D_train[:, -1].astype(np.int32)
        X_val = D_val[:, :-1]
        Y_val = D_val[:, -1].astype(np.int32)

        X_train_full = np.concatenate((X_train, X_val), axis=0)
        Y_train_full = np.concatenate((Y_train, Y_val))

        X_test = D_test[:, :-1]
        Y_test = D_test[:, -1].astype(np.int32)

        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train_full, Y_train_full,test_size=0.5)

        classes = np.unique(Y_train_full)
        n_classes = len(classes)
    #-------------------- Setting Pool classifiers -------------------------------------
        pool_classifiers = []
        if n_classifiers == 5:
            model_nb = GaussianNB().fit(X_train, y_train)
            model_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
            model_lr = LogisticRegression(solver = 'newton-cg').fit(X_train, y_train)
            model_xgboost = get_xgboost_classifier(n_classes).fit(X_train, y_train)
            model_rf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)

            pool_classifiers = [model_nb,model_knn,model_lr,model_xgboost,model_rf]

        # '''If pool_classifiers = None default RandomForestClassifier(n_estimators=200)'''
        elif n_classifiers == 0:
            pool_classifiers = None

        #------------------- KONARAE TRAINING --------------------------------------
        train_time_kne_start = time.time()
        kne = KNORAE(pool_classifiers)
        kne.fit(X_dsel, y_dsel)
        train_time_kne_end = time.time()

        #------------------- META DES TRAINING -------------------------------------
        train_time_mdes_start = time.time()
        meta = METADES(pool_classifiers)
        meta.fit(X_dsel, y_dsel)
        train_time_mdes_end = time.time()

        #------------------- KONARAU TRAINING --------------------------------------
        train_time_knu_start = time.time()
        knu = KNORAU(pool_classifiers)
        knu.fit(X_dsel, y_dsel)
        train_time_knu_end = time.time()
        
    #---------------------- Test KNE Pharse --------------------------
        test_time_kne_start = time.time()

        y_kne_pred = kne.predict(X_test)


        accuracy_kne = accuracy_score(Y_test, y_kne_pred)
        # print('accuracy = ', accuracy_kne)
        micro_f1_kne = f1_score(Y_test - 1, y_kne_pred - 1, average='micro')
        # print('micro_f1 =', micro_f1_kne)
        macro_f1_kne = f1_score(Y_test - 1, y_kne_pred - 1, average='macro')
        # print('macro_f1 =', macro_f1_kne)

        test_time_kne_end = time.time()

        #---------------------- Test META DES Pharse --------------------------
        test_time_mdes_start = time.time()

        y_mdes_pred = meta.predict(X_test)

        accuracy_mdes = accuracy_score(Y_test, y_mdes_pred)
        # print('accuracy = ', accuracy_mdes)
        micro_f1_mdes = f1_score(Y_test - 1, y_mdes_pred - 1, average='micro')
        # print('micro_f1 =', micro_f1_mdes)
        # print('support_macro:',precision_recall_fscore_support(Y_test, y_mdes_pred, average='macro'))
        macro_f1_mdes = f1_score(Y_test - 1, y_mdes_pred - 1, average='macro')
        # print('macro_f1 =', macro_f1_mdes)
        # print('micro:', precision_recall_fscore_support(Y_test, y_mdes_pred, average='micro'))
        test_time_mdes_end = time.time()

    #---------------------- Test KNU Pharse --------------------------
        test_time_knu_start = time.time()

        y_knu_pred = knu.predict(X_test)

        accuracy_knu = accuracy_score(Y_test, y_knu_pred)
        # print('accuracy = ', accuracy_knu)
        micro_f1_knu = f1_score(Y_test - 1, y_knu_pred - 1, average='micro')
        # print('micro_f1 =', micro_f1_knu)
        macro_f1_knu = f1_score(Y_test - 1, y_knu_pred - 1, average='macro')
        # print('macro_f1 =', macro_f1_knu)

        test_time_knu_end = time.time()

    #-------------------------------------- WRITE OUTPUT ---------------------------------------------
        result_kne = {'data':file_name,'n_classes': n_classes,'train_time':train_time_kne_end - train_time_kne_start,
        'test_time':test_time_kne_end - test_time_kne_start, 'accuracy':accuracy_kne,'micro_f1':micro_f1_kne, 'macro_f1':macro_f1_kne}
        meta_result_kne.append(result_kne)
        print('result_kne',accuracy_kne)

        result_mdes = {'data':file_name,'n_classes': n_classes,'train_time':train_time_mdes_end - train_time_mdes_start,
        'test_time':test_time_mdes_end - test_time_mdes_start, 'accuracy':accuracy_mdes,'micro_f1':micro_f1_mdes, 'macro_f1':macro_f1_mdes}
        meta_result_mdes.append(result_mdes)
        print('accuracy_mdes',accuracy_mdes)

        result_knu = {'data':file_name,'n_classes': n_classes,'train_time':train_time_knu_end - train_time_knu_start,
        'test_time':test_time_knu_end - test_time_knu_start, 'accuracy':accuracy_knu,'micro_f1':micro_f1_knu , 'macro_f1':macro_f1_knu}
        meta_result_knu.append(result_knu)
        print('accuracy_knu',accuracy_knu)

    except:
        print('===========File {}============='.format(file_name))
        meta_result_kne.append({'data':file_name,'n_classes': [], 'train_time': [], 'test_time': [], 'accuracy': [], 'micro_f1': [], 'macro_f1': []})
        meta_result_mdes.append({'data':file_name,'n_classes': [], 'train_time': [], 'test_time': [], 'accuracy': [], 'micro_f1': [], 'macro_f1': []})
        meta_result_knu.append({'data':file_name,'n_classes': [], 'train_time': [], 'test_time': [], 'accuracy': [], 'micro_f1': [], 'macro_f1': []})


if n_classifiers == 5:
    with open('result/result_knora-e(5).txt', 'w') as outfile:
        json.dump(meta_result_kne, outfile)

    with open('result/result_knora-u(5).txt', 'w') as outfile:
        json.dump(meta_result_knu, outfile)

    with open('result/result_meta-des(5).txt', 'w') as outfile:
        json.dump(meta_result_mdes, outfile)

elif n_classifiers == 0:
    with open('result/result_knora-e(rf200).txt', 'w') as outfile:
        json.dump(meta_result_kne, outfile)

    with open('result/result_knora-u(rf200).txt', 'w') as outfile:
        json.dump(meta_result_knu, outfile)

    with open('result/result_meta-des(rf200).txt', 'w') as outfile:
        json.dump(meta_result_mdes, outfile)