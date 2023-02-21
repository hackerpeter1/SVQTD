#! /usr/bin/env python3
import os
import pdb
import time
import json
import torch
import dataset
import argparse
import dataloader
import numpy as np
import torch.multiprocessing
import models as module_model
from utils.util import get_instance
from modules.head import FocalLoss
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')

# parameters
class_nums = [4, 4, 4, 2, 2, 3, 3, 3, 3, 3]  
class_weights = [torch.FloatTensor([22.8, 2.9, 1.0, 2.9]), torch.FloatTensor([12.7, 2.4, 1.0, 4.6]),
                torch.FloatTensor([1.0, 3.3, 7.5, 43.6]), torch.FloatTensor([1, 3.65]), torch.FloatTensor([1, 7.3]),
                torch.FloatTensor([2.7, 1, 19.1]), torch.FloatTensor([1.0, 1.36, 26.8]),
                torch.FloatTensor([1.0, 7.0, 10.3]), torch.FloatTensor([1.0, 3.9, 9.5]),
                torch.FloatTensor([1.0, 4.3, 22])]


def main(args, config):
    # dataset and dataloader
    train_dataset = get_instance(dataset, config['trainset'], num_task=config['num_task'])
    train_loader = get_instance(dataloader, config['trainloader'], dataset=train_dataset)
    validate_dataset = get_instance(dataset, config['validateset'], num_task=config['num_task'])
    validate_loader = get_instance(dataloader, config['validateloader'], dataset=validate_dataset)
    test_dataset = get_instance(dataset, config['testset'], num_task=config['num_task'])
    test_loader = get_instance(dataloader, config['testloader'], dataset=test_dataset)

    # model and history
    model = get_instance(module_model, config['model'], classes=class_nums[config['num_task']])
    model = torch.nn.DataParallel(model).cuda()
    print(model)
    print(args.start_epoch)
    if not os.path.exists('exp/' + args.save_name):
        print('mkdir %s' % ('exp/' + args.save_name))
        os.makedirs('exp/' + args.save_name)
    history = {'history': []}
    if args.start_epoch != 0:
        model.load_state_dict(torch.load('exp/%s/model_%s.pkl' % (args.model_name, args.start_epoch - 1)), map_location='cuda:0')
        print('loaded model ' + 'exp/%s/model_%s.pkl' % (args.model_name, args.start_epoch - 1))
        with open('exp/%s/history.json' % args.save_name, 'r') as stream:
            tmp = json.load(stream)
        history['history'] = tmp['history']

    # loss
    criterion = FocalLoss(gamma=2, class_weight=class_weights[config['num_task']]).cuda()

    # optimizer and scheduler
    optimizer = get_instance(torch.optim, config['optimizer'], model.parameters())
    lr_scheduler = get_instance(torch.optim.lr_scheduler, config['lr_scheduler'], optimizer)


    min_epoch = 10 # at least train 10 epochs
    for epoch in range(args.start_epoch, args.start_epoch + config['epochs']):
        # preparation
        train_losses = AverageMeter()
        train_top1 = AverageMeter()
        validate_losses = AverageMeter()
        validate_top1 = AverageMeter()
        test_losses = AverageMeter()
        test_top1 = AverageMeter()

        train_uar = train(model, epoch, optimizer, criterion, train_loader, train_losses, train_top1)
        valid_uar = validate(model, epoch, criterion, validate_loader, validate_losses, validate_top1)
        test_uar = validate(model, epoch, criterion, test_loader, test_losses, test_top1)

        lr_scheduler.step()
        

        history_block = {
            'model_epoch': epoch,
            'train_loss': train_losses.avg.data.cpu().numpy().tolist(),
            'validate_loss': validate_losses.avg.data.cpu().numpy().tolist(),
            'test_loss': test_losses.avg.data.cpu().numpy().tolist(),
            'train_acc': train_top1.avg.data.cpu().numpy().tolist(),
            'valid_acc': validate_top1.avg.data.cpu().numpy().tolist(),
            'test_acc': test_top1.avg.data.cpu().numpy().tolist(),
            'train_uar': train_uar,
            'valid_uar': valid_uar,
            'test_uar': test_uar
        }
        history['history'].append(history_block)

        with open('exp/%s/history.json' % args.save_name, 'w') as outfile:
            outfile.write(json.dumps(history, indent=4))

        min_epoch = min_epoch - 1
        if (min_epoch <= 0):
            if (train_losses.avg.data.cpu().numpy() < validate_losses.avg.data.cpu().numpy()):
                print("Early stop to avoid the overfitting.")
                break


def train(model, epoch, optimizer, criterion, dataloader, losses, top1):
    end = time.time()
    y_true = []
    y_pred = []
    model.train()
    for i, data in enumerate(dataloader):
        # measure data loading time
        data_time = time.time() - end

        # prepare input and label
        feat = data['data'].cuda()
        labels = data['label'].cuda()  # ex. [2,3,1,2,2,2,0]
        # masks = data['data_mask'].cuda()
        # compute output and loss
        optimizer.zero_grad()
        output = model(feat.transpose(1, 2))
        loss = criterion(output['out'], labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        y_true.append(data['label'].data.cpu())
        y_pred.append(torch.max(output['out'], dim=1)[1].data.cpu())

        # print output and label
        output_top1 = torch.max(output['out'], dim=1)[1]
        end = time.time()
        # print(output_top1)
        # print(labels)

        # measure accuracy and record loss
        precision = ((output_top1 == labels).sum().float() / (feat.size()[0] * 1)) * 100
        losses.update(loss.data, feat.size()[0])
        top1.update(precision.data, feat.size()[0])

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()

        print('train:\tlength [%d]\t'
              'epoch [%d][%d/%d]\t'
              'Time [%.3f/%.3f]\t'
              'Loss %.4f %.4f\t'
              'Accuracy %.2f %.2f\t'
              % (
                  feat.size()[2], epoch, i + 1, len(dataloader), batch_time,
                  data_time, losses.val, losses.avg, top1.val, top1.avg))

    # save model
    save_model_name = 'exp/%s/model_%s.pkl' % (args.save_name, epoch)
    torch.save(model.state_dict(), save_model_name)

    # training UAR metric and save model
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    result = str(recall_score(y_true, y_pred, average='macro'))
    print("Train UAR | Epoch {0}: {1}".format(epoch, result))
    return result


def validate(model, epoch, criterion, dataloader, losses, top1):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # prepare input and label
            feat = data['data'].cuda()
            labels = data['label'].cuda()
            # masks = data['data_mask'].cuda()
            # compute output and loss
            print('train:')
            output = model(feat.transpose(1, 2))
            loss = criterion(output['out'], labels)

            y_true.append(data['label'])
            y_pred.append(torch.max(output['out'], dim=1)[1].data.cpu())

            # precision
            output_top1 = torch.max(output['out'], dim=1)[1]
            precision = ((output_top1 == labels).sum().float() / (feat.size()[0] * 1)) * 100
            losses.update(loss.data, feat.size()[0])
            top1.update(precision.data, feat.size()[0])

            print('Validation\tLength [%d]\t'
                  'Epoch [%d][%d/%d]\t'
                  'Accuracy %.2f %.2f\t'
                  % (
                      feat.size()[2], epoch, i + 1, len(dataloader),
                      top1.val, top1.avg))

    # UAR metric
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    result = str(recall_score(y_true, y_pred, average='macro'))
    print("Validation UAR | Epoch {0}: {1}".format(epoch, result))
    return result


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size()(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Paralinguistic Singing Attribute Recognition-E2E')
    parser.add_argument('-c', '--config', default=None, type=str, required=True,
                        help='config file path (default: None)')
    parser.add_argument('--start-epoch', default=0, type=int, required=True,
                        help='start epoch. if start-epoch>0 will resume training.  (default: 0)')
    parser.add_argument('--model-name', default=None, type=str, required=True,
                        help='load the E2E model by this name from ./exp/. (default: None)')
    parser.add_argument('--save-name', default=None, type=str, required=True,
                        help='save the E2E model by this name from ./exp/. (default: None)')
    args = parser.parse_args()
    with open(args.config) as rfile:
        config = json.load(rfile)

    main(args, config)
