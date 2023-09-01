import numpy as np
import pandas as pd
import torch
from utils.utils import *
from utils.sampling_utils import generate_sample_idxs, generate_features_array, update_sampling_weights
from utils.core_utils import train_loop, train_loop_clam,validate,validate_clam
from datasets.dataset_generic import Generic_MIL_Dataset
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.neighbors import NearestNeighbors
#import scann
from ray import tune
import random 

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, min_epochs=20, patience=20, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.min_epochs = min_epochs

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss
        
        if epoch >= self.min_epochs:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
            elif score < self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name, True)
                self.counter = 0
        else:
                self.save_checkpoint(val_loss, model, ckpt_name, False)

    def save_checkpoint(self, val_loss, model, ckpt_name, better_model=True):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if better_model:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(f'Below min epochs. Validation loss changed ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train_sampling(config,datasets, cur, class_counts, args):
    """   
        train for a single fold
    """
    assert 0<=args.sampling_random<=1,"sampling_random needs to be between 0 and 1"

    if args.tuning:
        args.lr=config["lr"]
        args.reg=config["reg"]
        args.drop_out=config["drop_out"]
        if not args.no_inst_cluster:
            args.B=config["B"]
        args.no_sampling_epochs=config["no_sample"]
        args.weight_smoothing=config["weight_smoothing"]
        args.resampling_iterations=config["resampling_iterations"]
        args.samples_per_iteration=int(960/(config["resampling_iterations"]))
        args.sampling_neighbors=config["sampling_neighbors"]
        args.sampling_random=config["sampling_random"]
        args.sampling_random_delta=config["sampling_random_delta"]
        while args.B>args.samples_per_iteration:
            args.B=int(args.B/2)
            print("args.B reduced to {} due to samples_per_iteration being too small ({})".format(args.B,args.samples_per_iteration))

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'balanced_ce':
        ce_weights=[(1/class_counts[i])*(sum(class_counts)/len(class_counts)) for i in range(len(class_counts))]
        print("weighting cross entropy with weights {}".format(ce_weights))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights).to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()
    if args.continue_training:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_split.load_from_h5(False)
    train_loader = get_split_loader(train_split, training=True, weighted = args.weighted_sample)
    train_split_h5=train_split
    train_split_h5.load_from_h5(True)
    train_loader_h5 = get_split_loader(train_split_h5, training=True, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping and not args.tuning:
        early_stopping = EarlyStopping(min_epochs = args.min_epochs, patience = 50, stop_epoch=50, verbose = True)
    else:
        early_stopping = None
    print('Done!')

    
    train_loader_nbrs = get_split_loader(train_split_h5, training=True, testing = False, weighted = False)
    ## prepare the nbrs calls
    args.neighbors_prep=False
    args.neighbors_function="ball_tree"
    assert args.neighbors_function in ["ball_tree","scann_ah"]
    print("need to move args.neighbors_prep to proper place and make scann and brute options")
    nbrs_dict={}
    X_dict={}
    if args.neighbors_prep:
        raise NotImplementedError("plan to try calculating all neighbors for all slides before training to reduce repeated computations")
        for batch_idx, (data, label,coords,slide_id) in enumerate(train_loader_nbrs):
            coords=torch.tensor(coords)
            X = generate_features_array(args, data, coords, slide_id, slide_id_list=[],texture_dataset=[])
            if args.neighbors_function=="ball_tree":
                nbrs_dict[slide_id[0][0]] = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
            elif args.neighbors_function=="scann_ah":
                nbrs_dict[slide_id[0][0]]=scann.scann_ops_pybind.builder(X,args.sampling_neighbors,"dot_product").score_ah(2).build()
            X_dict[slide_id[0][0]] = X   
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            assert args.samples_per_iteration>=args.B, "B too large for sampling"
            assert args.final_sample_size>=args.B, "B too large for final sample"
            if epoch<args.no_sampling_epochs:
                train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
                stop, val_error, val_loss,val_auc = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                    early_stopping, writer, loss_fn, args.results_dir)
            else:
                train_loop_clam_sampling(epoch, model, train_loader_h5, optimizer, args.n_classes, args.bag_weight, args, writer, loss_fn, nbrs_dict, X_dict)
                stop, val_error, val_loss,val_auc = validate_clam_sampling(cur, epoch, model, val_loader, args.n_classes,
                    early_stopping, writer, loss_fn, args.results_dir)
        else:
            if epoch<args.no_sampling_epochs:
                train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
                stop, val_error, val_loss,val_auc = validate(cur, epoch, model, val_loader, args.n_classes, 
                    early_stopping, writer, loss_fn, args.results_dir)
            else:
                train_loop_sampling(epoch, model, train_loader_h5, optimizer, args.n_classes, args, writer, loss_fn)
                stop, val_error, val_loss, val_auc = validate_sampling(cur, epoch, model, val_loader, args.n_classes,
                    early_stopping, writer, loss_fn, args.results_dir)
        
        if args.tuning:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=val_loss, accuracy=1-val_error, auc=val_auc)

        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam_sampling(epoch, model, loader, optimizer, n_classes, bag_weight, args, writer = None, loss_fn = None, nbrs_dict = None, X_dict = None):
    num_random=int(args.samples_per_iteration*args.sampling_random)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    if args.sampling_average:
        sampling_update='average'
    else:
        sampling_update='max'
    
    ## Collecting Y_hats and labels to view performance across sampling iterations
    Y_hats=[]
    labels=[]
    Y_probs=[]
    all_logits=[]
    slide_ids = loader.dataset.slide_data['slide_id']

    slide_id_list=[]
    texture_dataset=[]
    if args.sampling_type=='textural':
        if args.texture_model=='levit_128s':
            texture_dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                        data_dir= os.path.join(args.data_root_dir, 'levit_128s'),
                        shuffle = False,
                        print_info = True,
                        label_dict = args.label_dict,
                        patient_strat= False,
                        ignore=[])
            slide_id_list = list(pd.read_csv(args.csv_path)['slide_id'])

    print('\n')

    total_samples_per_slide = (args.samples_per_iteration*args.resampling_iterations)+args.final_sample_size
    print("Total patches sampled per slide: ",total_samples_per_slide)
    for batch_idx, (data, label,coords,slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        if args.fully_random or total_samples_per_slide>=len(coords):
            if total_samples_per_slide>=len(coords):    
                print("full slide used for slide {} with {} patches".format(slide_id,len(coords)))
                data_sample=data
            else:
                sample_idxs=list(np.random.choice(range(0,len(coords)), size=total_samples_per_slide,replace=False))
                all_sample_idxs=sample_idxs
                data_sample=data[sample_idxs]#.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data_sample, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            train_loss += loss_value
            
            instance_loss = instance_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            error = calculate_error(Y_hat, label)
            train_error += error

            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss

            # backward pass
            total_loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()
            continue

        coords=torch.tensor(coords)
        if args.neighbors_prep:
            X = X_dict[slide_id[0][0]]
        else:
            X = generate_features_array(args, data, coords, slide_id, slide_id_list, texture_dataset)

        ## First sampling iteration (fully random sampling)
        sample_idxs=list(np.random.choice(range(0,len(coords)), size=args.samples_per_iteration,replace=False))
        all_sample_idxs=sample_idxs
        data_sample=data[sample_idxs]#.to(device)
        with torch.no_grad():
            logits, Y_prob, Y_hat, raw_attention, _ = model(data_sample, label=label, instance_eval=True)
        
        attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0]#.cpu()

        Y_hats.append(Y_hat)
        labels.append(label)
        Y_probs.append(Y_prob)
        all_logits.append(logits)
        
        if args.neighbors_prep:
            nbrs=nbrs_dict[slide_id[0][0]]
        else:
            if args.neighbors_function=='scann_ah':
                nbrs = scann.scann_ops_pybind.builder(X,args.sampling_neighbors,"dot_product").score_ah(2).build()
                indices, distances = nbrs.search_batched(X[sample_idxs], final_num_neighbors=args.sampling_neighbors)
            elif args.neighbors_function=='ball_tree':
                nbrs = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
                distances, indices = nbrs.kneighbors(X[sample_idxs])
        
        if args.neighbors_function=='scann_ah':
            indices, distances = nbrs.search_batched(X[sample_idxs], final_num_neighbors=args.sampling_neighbors)
        elif args.neighbors_function=='ball_tree':
            distances, indices = nbrs.kneighbors(X[sample_idxs])
        
        sampling_random=args.sampling_random

        ## Subsequent sampling iterations
        neighbors=args.sampling_neighbors
        sampling_weights=np.full(shape=len(coords),fill_value=0.0001)
        for iteration_count in range(args.resampling_iterations-2):
            if sampling_random>args.sampling_random_delta:
                sampling_random=sampling_random-args.sampling_random_delta
            else:
                sampling_random=0
            
            num_random=int(args.samples_per_iteration*sampling_random)
            
            sampling_weights = update_sampling_weights(sampling_weights, attention_scores, all_sample_idxs, indices, neighbors, power=args.weight_smoothing, normalise = False, sampling_update=sampling_update, repeats_allowed = False)
            sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),args.samples_per_iteration,num_random)
            all_sample_idxs=all_sample_idxs+sample_idxs
            
            if args.neighbors_function=='scann_ah':
                indices, distances = nbrs.search_batched(X[sample_idxs], final_num_neighbors=args.sampling_neighbors)
            elif args.neighbors_function=='ball_tree':
                distances, indices = nbrs.kneighbors(X[sample_idxs])
            #data_sample=data[sample_idxs].to(device)
            data_sample=data[sample_idxs]

            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, _ = model(data_sample)
            attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0]#.cpu()

        ## final sample
        num_random=int(args.samples_per_iteration*sampling_random)
        sampling_weights = update_sampling_weights(sampling_weights, attention_scores, all_sample_idxs, indices, neighbors, power=args.weight_smoothing, normalise = False, sampling_update=sampling_update, repeats_allowed = False)
        sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),args.samples_per_iteration,num_random)
        all_sample_idxs=all_sample_idxs+sample_idxs
        if args.use_all_samples:
            for sample_idx in all_sample_idxs:
                sampling_weights[sample_idx]=0
            sample_idxs=list(np.random.choice(range(0,len(coords)),p=sampling_weights/sum(sampling_weights),size=int(args.final_sample_size),replace=False))
            all_sample_idxs=all_sample_idxs+sample_idxs
            data_sample=data[all_sample_idxs]#.to(device)
        else:
            raise NotImplementedError("Only implemented with --use_all_samples at the minute")

        logits, Y_prob, Y_hat, _, instance_dict = model(data_sample,label=label,instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        instance_loss = instance_dict['instance_loss']
        
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop_sampling(epoch, model, loader, optimizer, n_classes, args, writer = None, loss_fn = None):   
    num_random=int(args.samples_per_iteration*args.sampling_random)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    
    if args.sampling_average:
        sampling_update='average'
    else:
        sampling_update='max'

    ## Collecting Y_hats and labels to view performance across sampling iterations
    #Y_hats=[]
    #labels=[]
    #Y_probs=[]
    #all_logits=[]
    slide_ids = loader.dataset.slide_data['slide_id']

    slide_id_list=[]
    texture_dataset=[]
    if args.sampling_type=='textural':
        if args.texture_model=='levit_128s':
            texture_dataset =  Generic_MIL_Dataset(csv_path = args.csv_path,
                data_dir= os.path.join(args.data_root_dir, 'levit_128s'),
                shuffle = False,
                print_info = True,
                label_dict = args.label_dict,
                patient_strat= False,
                ignore=[])
            slide_id_list = list(pd.read_csv(args.csv_path)['slide_id'])
        
    print('\n')

    total_samples_per_slide = (args.samples_per_iteration*args.resampling_iterations)+args.final_sample_size
    print("Total patches sampled per slide: ",total_samples_per_slide)
    for batch_idx, (data, label,coords,slide_id) in enumerate(loader):
        coords=torch.tensor(coords)
        #data, label = data.to(device), label.to(device)
        #data = data.to(device)
        label = label.to(device)
        if args.fully_random or total_samples_per_slide>=len(coords):
            if total_samples_per_slide>=len(coords):
                print("full slide used for slide {} with {} patches".format(slide_id,len(coords)))
                data_sample=data
            else:
                sample_idxs=list(np.random.choice(range(0,len(coords)), size=total_samples_per_slide,replace=False))
                all_sample_idxs=sample_idxs
                data_sample=data[sample_idxs]#.to(device)
            data_sample = data_sample.to(device)
            logits, Y_prob, Y_hat, _, _ = model(data_sample)

            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            train_loss += loss_value
            error = calculate_error(Y_hat, label)
            train_error += error

            # backward pass
            loss.backward()
            # step
            optimizer.step()
            continue


        X = generate_features_array(args, data, coords, slide_id, slide_id_list, texture_dataset)

        ## First sampling iteration (fully random sampling)
        #sample_idxs=list(np.random.choice(range(0,len(coords)), size=args.samples_per_iteration,replace=False))
        sample_idxs=list(random.sample(range(len(coords)),args.samples_per_iteration))
        all_sample_idxs=sample_idxs
        data_sample=data[sample_idxs].to(device)
        with torch.no_grad():
            logits, Y_prob, Y_hat, raw_attention, _ = model(data_sample)
        
        attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0]#.cpu()

        #Y_hats.append(Y_hat)
        #labels.append(label)
        #Y_probs.append(Y_prob)
        #all_logits.append(logits)
                                                 
        ## Find nearest neighbors of each patch to prepare for spatial resampling
        nbrs = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X[sample_idxs])
        
        ## Subsequent sampling iterations
        sampling_random=args.sampling_random
        neighbors=args.sampling_neighbors
        sampling_weights=np.full(shape=len(coords),fill_value=0.001)

        for iteration_count in range(args.resampling_iterations-2):
            if sampling_random>args.sampling_random_delta:
                sampling_random=sampling_random-args.sampling_random_delta
            else:
                sampling_random=0
            
            num_random=int(args.samples_per_iteration*sampling_random)
            
            sampling_weights = update_sampling_weights(sampling_weights, attention_scores, all_sample_idxs, indices, neighbors, power=args.weight_smoothing, normalise = False, sampling_update=sampling_update, repeats_allowed = False)
            sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),args.samples_per_iteration,num_random)
            all_sample_idxs=all_sample_idxs+sample_idxs
            distances, indices = nbrs.kneighbors(X[sample_idxs])
            data_sample=data[sample_idxs].to(device)
            
            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, _ = model(data_sample)
            attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0]#.cpu()
        
        ## final sample
        num_random=int(args.samples_per_iteration*sampling_random)
        sampling_weights = update_sampling_weights(sampling_weights, attention_scores, all_sample_idxs, indices, neighbors, power=args.weight_smoothing, normalise = False, sampling_update=sampling_update, repeats_allowed = False)
        sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),args.samples_per_iteration,num_random)
        all_sample_idxs=all_sample_idxs+sample_idxs
        if args.use_all_samples:
            for sample_idx in all_sample_idxs:
                sampling_weights[sample_idx]=0
            sample_idxs=list(np.random.choice(range(0,len(coords)),p=sampling_weights/sum(sampling_weights),size=int(args.final_sample_size),replace=False))
            all_sample_idxs=all_sample_idxs+sample_idxs
            data_sample=data[all_sample_idxs].to(device)
        else:
            raise NotImplementedError("Only implemented with --use_all_samples at the minute")
        
        logits, Y_prob, Y_hat, _, _ = model(data_sample)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        
   
def validate_sampling(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        #early_stopping(epoch, 1-auc, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            with open(os.path.join(results_dir,'early_stopping{}.txt'.format(cur)), 'w') as f:
                f.write('Finished at epoch {}'.format(epoch))
            print("Early stopping")
            return True, val_error, val_loss, auc

    return False, val_error, val_loss, auc

def validate_clam_sampling(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        #early_stopping(epoch, 1-auc, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            with open(os.path.join(results_dir,'early_stopping{}.txt'.format(cur)), 'w') as f:
                f.write('Finished at epoch {}'.format(epoch))
            print("Early stopping")
            return True, val_error, val_loss, auc

    return False, val_error, val_loss, auc

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return test_error, auc, acc_logger
