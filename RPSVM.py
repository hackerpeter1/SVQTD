#! /usr/bin/env python3
import pdb
import json
import torch
import pickle
import dataset
import argparse
import dataloader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import svm
from utils.util import *
import models as module_model
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, recall_score, confusion_matrix

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')

# parameters
class_nums = [4,4,4,2,2,3,3,3,3,3]
complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]
show_confusion = True                             

def main(args, config):
    # init dataset and dataloader
    train_dataset = get_instance(dataset, config['trainset'], num_task=config['num_task'])
    train_dataloader = get_instance(dataloader, config['trainloader'], dataset=train_dataset)
    dev_dataset = get_instance(dataset, config['validateset'], num_task=config['num_task'])
    dev_dataloader = get_instance(dataloader, config['validateloader'], dataset=dev_dataset)
    test_dataset = get_instance(dataset, config['testset'], num_task=config['num_task'])
    test_dataloader = get_instance(dataloader, config['testloader'], dataset=test_dataset)


    # load model to extract embedding online
    print('Load: exp/%s/model_%s.pkl' % (config["E2Emodel_name"], config["E2Emodel_epoch"]-1))
    model = get_instance(module_model, config['model'], classes=class_nums[config['num_task']])
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load('exp/%s/model_%s.pkl' % (config["E2Emodel_name"], config["E2Emodel_epoch"]-1), map_location='cuda:0')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # extract train dev test embedding and label for each task
    print('Get train embedding and label...')
    X_train = [] 
    Y_train = []
    for i, data in enumerate(tqdm(train_dataloader)):
        input_x = Variable(data['data'].cuda())
        input_x = input_x.permute(0,2,1)
        output = model(input_x)
        X_train.append(output['embd'].detach().cpu())
        Y_train.append(data['label'])
    X_train = torch.cat(X_train, dim=0).numpy()
    Y_train = torch.cat(Y_train, dim=0).squeeze().numpy()
    
    print('Get dev embedding and label...')
    X_dev = [] 
    Y_dev = []
    for i, data in enumerate(tqdm(dev_dataloader)):
        input_x = Variable(data['data'].cuda())
        input_x = input_x.permute(0,2,1)
        output = model(input_x)
        X_dev.append(output['embd'].detach().cpu())
        Y_dev.append(data['label'])
    X_dev = torch.cat(X_dev, dim=0).numpy()
    Y_dev = torch.cat(Y_dev, dim=0).squeeze().numpy()
    
    print('Get test embedding and label...')
    X_test = [] 
    Y_test = []
    info_test = []
    for i, data in enumerate(tqdm(test_dataloader)):
        input_x = Variable(data['data'].cuda())
        input_x = input_x.permute(0,2,1)
        output = model(input_x)
        X_test.append(output['embd'].detach().cpu())
        Y_test.append(data['label'])
        info_test.append(data['filename'])
    X_test = torch.cat(X_test, dim=0).numpy()
    Y_test = torch.cat(Y_test, dim=0).squeeze().numpy()
    info_test = np.concatenate(info_test, axis=0)
    
    # Train SVM model with different complexities and evaluate
    print('Train...')
    uar_scores = [] 
    results = []
    optim_comp = -1
    max_UAR = -1
    for comp in complexities:
        print('\ncomplexity {0:.6f}'.format(comp))
        clf = make_pipeline(StandardScaler(), svm.LinearSVC(C=comp, class_weight='balanced', random_state=0, max_iter=100000))
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_dev)
        current_UAR = recall_score(Y_dev, Y_pred, average='macro')
        if current_UAR > max_UAR:
            optim_comp = comp
            max_UAR = current_UAR
        if show_confusion:
            print('confusion matrix (dev):')
            print(confusion_matrix(Y_dev, Y_pred))
    
    X_traindev = np.concatenate((X_train, X_dev))
    Y_traindev = np.concatenate((Y_train, Y_dev))
    final_clf = make_pipeline(StandardScaler(), svm.LinearSVC(C=optim_comp, class_weight='balanced', random_state=0, max_iter=1000000))
    final_clf.fit(X_traindev, Y_traindev)
    Y_pred = final_clf.predict(X_test)

    final_clf_2 = make_pipeline(StandardScaler(), svm.LinearSVC(C=optim_comp, class_weight='balanced', random_state=0, max_iter=1000000))
    final_clf_2.fit(X_train, Y_train)
    Y_pred_2 = final_clf_2.predict(X_test)

    if not os.path.exists(os.path.join(args.save_path, config['E2Emodel_name'])):
        # save train model 
        os.makedirs(os.path.join(args.save_path, config['E2Emodel_name']))
        pickle.dump(final_clf, open(os.path.join(args.save_path, config['E2Emodel_name'], 'traindev_model'), 'wb'))
        pickle.dump(final_clf_2, open(os.path.join(args.save_path, config['E2Emodel_name'], 'train_model'), 'wb'))

    if not os.path.exists(os.path.join(args.save_path, config['E2Emodel_name'], 'result')):
        # save UAR result
        os.makedirs(os.path.join(args.save_path, config['E2Emodel_name'], 'result'))
    result = classification_report(Y_test, Y_pred)
    result_2 = classification_report(Y_test, Y_pred_2)
    f = open(os.path.join(args.save_path, config['E2Emodel_name'], 'result', 'UAR.txt'), "w+")
    f.write('traindev:\n')
    f.write(str(result))
    f.write('\n\n')
    f.write('train:\n')
    f.write(str(result_2))
    f.close()
    # save Y_pred to csv
    pred_csv = pd.concat([pd.DataFrame(info_test), pd.DataFrame(Y_pred), pd.DataFrame(Y_test)], axis=1)
    pred_csv.columns = ['path', 'pred', 'true']
    pred_csv.to_csv(os.path.join(args.save_path, config['E2Emodel_name'], 'result', 'pred_csv_traindev.csv'), index=False)

    pred_csv_2 = pd.concat([pd.DataFrame(info_test), pd.DataFrame(Y_pred_2), pd.DataFrame(Y_test)], axis=1)
    pred_csv_2.columns = ['path', 'pred', 'true']
    pred_csv_2.to_csv(os.path.join(args.save_path, config['E2Emodel_name'], 'result', 'pred_csv_train.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Paralinguistic Singing Attribute Recognition-RPSVM')
    parser.add_argument('-c', '--config', default=None, type=str, required=True,
                        help='config file path (default: None)')
    parser.add_argument('--save-path', default="../model/RPSVM", type=str,
                        help='save the RPSVM model by given path. (default: ../model/RPSVM)')
    args = parser.parse_args()
    with open(args.config) as rfile:
        config = json.load(rfile)

    main(args, config)
