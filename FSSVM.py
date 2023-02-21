import os
import pdb
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import date
from tqdm import tqdm
from utils.util import *
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

def train(args):
    print('Reading train csv...')
    Y_train_csv = pd.read_csv(args.train_csv).sample(frac=1)
    X_train_filenames = get_feature_filenames(Y_train_csv)
    Y_train = Y_train_csv.drop(columns={"name","num","song"},axis=1).values
    print('Loading train features...')
    X_train = get_features(args, X_train_filenames) 

    print('Reading dev csv...')
    Y_dev_csv = pd.read_csv(args.dev_csv).sample(frac=1)
    X_dev_filenames = get_feature_filenames(Y_dev_csv)
    Y_dev = Y_dev_csv.drop(columns={"name","num","song"},axis=1).values
    print('Loading dev features...')
    X_dev = get_features(args, X_dev_filenames) 


    #X_traindev = np.concatenate((X_train, X_dev))
    #Y_traindev = np.concatenate((Y_train, Y_dev))

    task_num = Y_train.shape[1]
    complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    classifiers = []
    for i in range(task_num):
        classifiers.append([])
        max_comp = -1
        max_UAR = -1
        for comp in complexities:
            # find best comp
            classifier = make_pipeline(StandardScaler(), LinearSVC(C = comp, class_weight = 'balanced', random_state=0, max_iter=100000))
            classifier.fit(X_train, Y_train[:,i])
            Y_pred = classifier.predict(X_dev)
            current_UAR = recall_score(Y_dev[:,i], Y_pred, average='macro')
            if current_UAR > max_UAR:
                max_comp = comp
                max_UAR = current_UAR
            classifiers[i].append(classifier)
        final_classifier = make_pipeline(StandardScaler(), LinearSVC(C = max_comp, class_weight = 'balanced', random_state=0, max_iter=100000))
        final_classifier.fit(X_train, Y_train[:,i])
        # save model
        if not os.path.exists(os.path.join(args.model_dir)):
            os.makedirs(os.path.join(args.model_dir))
        pickle.dump(final_classifier, open(os.path.join(args.model_dir, "classifier_task" + str(i)), 'wb'))


def test(args):
    print('Reading test csv...')
    Y_test_csv = pd.read_csv(args.test_csv).sample(frac=1)
    Y_test_info = Y_test_csv[["name", "num", "song"]]
    X_test_filenames = get_feature_filenames(Y_test_csv)
    Y_test = Y_test_csv.drop(columns={"name","num","song"},axis=1).values
    print('Loading test features...')
    X_test = get_features(args, X_test_filenames) 

    task_num = Y_test.shape[1]
    UARs = []
    for i in range(task_num):
        # load trained model
        classifier = pickle.load(open(os.path.join(args.model_dir, 'classifier_task' + str(i)), 'rb'))
        Y_pred = classifier.predict(X_test)
        UARs.append(recall_score(Y_test[:,i], Y_pred, average='macro'))
        # save pred result to csv for each task
        pred_csv = pd.concat([Y_test_info, pd.DataFrame(Y_pred), pd.DataFrame(Y_test)[i]], axis=1)
        pred_csv.columns = ['name', 'num', 'song', 'pred', 'true']
        if not os.path.exists(os.path.join(args.model_dir, 'result')):
            os.makedirs(os.path.join(args.model_dir, 'result'))
        pred_csv.to_csv(os.path.join(args.model_dir, 'result', 'pred_csv' + str(i) + ".csv"), index=False)
        
    with open(os.path.join(args.model_dir, 'result', 'UARs.txt'), "w+") as UAR_file:
        for i in range(task_num):        
            UAR_file.write(str(UARs[i]))
            UAR_file.write('\n')


def main():
    parser = argparse.ArgumentParser('Paralinguistic singing attribute recognition-FSSVM')
    parser.add_argument('--is-train', default=True, type=str2bool, required=True,
                        help='True for train mode, false for test mode.')
    parser.add_argument('--train-csv', default='./csv/train.csv', type=str,
                        help='In train mode, you should propose the path of the train csv and the dev csv.\n' + 
                             'In test mode, you should only propose the path of test csv.')

    parser.add_argument('--dev-csv', default='./csv/dev.csv', type=str,
                        help='In train mode, you should propose the path of the train csv and the dev csv.\n' + 
                             'In test mode, you should only propose the path of test csv.')
    parser.add_argument('--test-csv', default='./csv/test.csv', type=str,
                        help='In train mode, you should propose the path of the train csv and the dev csv.\n' + 
                             'In test mode, you should only propose the path of test csv.')
    parser.add_argument('--feature-dir', default="./IS09_features", type=str,
                        help='The path of the features for train or test.')
    parser.add_argument('--model-dir', default="./model/FSSVM", type=str,
                        help='The path of the FSSVM model to save for train model, and to load for test model.')
    args = parser.parse_args()
    if args.is_train: 
        train(args)
    else: 
        test(args)

if __name__ == "__main__":
    main()


