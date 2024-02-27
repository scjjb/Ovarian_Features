import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_graph import Graph_Model
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.resnet_custom import resnet18_baseline,resnet50_baseline
import timm
import pandas as pd
from ray import tune
import warnings
## We can't use the latest version of torch-scatter on this version of pytorch, which it keeps warning us to do to increase speed
warnings.simplefilter(action='ignore', category=UserWarning)

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
    def __init__(self, min_epochs=50, patience=50, stop_epoch=50, verbose=False):
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

def train(config, datasets, cur, class_counts_train, class_counts_val, args):
    """   
        train for a single fold
    """
    ## If tuning, update args from config 
    if args.tuning:
        if args.model_type in ["graph","graph_ms"]:
            args.graph_edge_distance=config["edge_distance"]
            args.pooling_layers=config["poolings"]
            args.message_passings=config["passings"]
            args.gat_heads=config["gat_heads"]
            args.pooling_factor=config["pooling_factor"]
            try:
                args.embedding_size=config["A_embedding_size"]
            except:
                args.embedding_size=config["embedding_size"]
        
        else:
            if not args.no_inst_cluster:
                args.B=config["B"]
            try:
                args.model_size=config["A_model_size"]
            except:
                args.model_size=config["model_size"]
        

        args.lr=config["lr"]
        args.beta1=config["beta1"]
        args.beta2=config["beta2"]
        args.eps=config["eps"]
        args.reg=config["reg"]
        args.drop_out=config["drop_out"]
        args.lr_factor=config["lr_factor"]
        args.lr_patience=config["lr_patience"]
        try:
            args.max_patches_per_slide=config["patches"]
        except:
            args.max_patches_per_slide=config["A_patches"]


    if args.extract_features:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('loading {} pretrained model {}'.format(args.pretraining_dataset, args.model_architecture))
        feature_extractor_model=None
        if args.model_architecture=='resnet18':
            feature_extractor_model = resnet18_baseline(pretrained=True,dataset=args.pretraining_dataset)
        elif args.model_architecture=='resnet50':
            feature_extractor_model = resnet50_baseline(pretrained=True,dataset=args.pretraining_dataset)
        elif args.model_architecture=='levit_128s':
            feature_extractor_model = timm.create_model('levit_256',pretrained=True, num_classes=0)
        else:
            raise NotImplementedError("this model_architecture is not implemented for feature extraction during training")
        feature_extractor_model.to(device)
    else:
        feature_extractor_model = None

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=60)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    
    train_split.max_patches_per_slide=args.max_patches_per_slide
    val_split.max_patches_per_slide=float('inf')
    test_split.max_patches_per_slide=float('inf')
    print("\nTraining max patches per slide: ",train_split.max_patches_per_slide)
    print("Validation max patches per slide: ",val_split.max_patches_per_slide)
    print("Testing max patches per slide: ",test_split.max_patches_per_slide)
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.to(device,non_blocking=True)
        loss_fn_val = loss_fn
    elif args.bag_loss == 'balanced_ce':
        ce_weights=[(1/class_counts_train[i])*(sum(class_counts_train)/len(class_counts_train)) for i in range(len(class_counts_train))]
        print("weighting training cross entropy with weights {}".format(ce_weights))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights).to(device,non_blocking=True)).to(device,non_blocking=True)
        
        ce_weights_val=[(1/class_counts_val[i])*(sum(class_counts_val)/len(class_counts_val)) for i in range(len(class_counts_val))]
        print("weighting training cross entropy with weights {}".format(ce_weights_val))
        loss_fn_val = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights_val).to(device,non_blocking=True)).to(device,non_blocking=True)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device,non_blocking=True)
        loss_fn_val = nn.CrossEntropyLoss().to(device,non_blocking=True)


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
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
    
    elif args.model_type in ['graph','graph_ms']:
        model = Graph_Model(pooling_factor=args.pooling_factor, pooling_layers=args.pooling_layers, message_passings=args.message_passings, gat_heads=args.gat_heads, embedding_size=args.embedding_size ,num_features=train_split[0][0].shape[1], num_classes=args.n_classes,drop_out=args.drop_out, message_passing=args.message_passing, pooling=args.pooling)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    print("\nModel parameters:",f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, non_blocking=True)
    if args.continue_training:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...\n')
    if args.plot_graph == "seperate":
        max_epochs=1
        args.debug_loader=True

    if args.debug_loader:
        train_split.set_debug_loader(True)
        val_split.set_debug_loader(True)
        test_split.set_debug_loader(True)
    
    ## update graph edge distance 
    if args.tuning:
        train_split.graph_edge_distance = args.graph_edge_distance
        val_split.graph_edge_distance = args.graph_edge_distance
        test_split.graph_edge_distance = args.graph_edge_distance

    if args.extract_features:
        train_split.set_extract_features(True)
    if args.augment_features:
        train_split.set_augment_features(True)
    train_split.set_transforms()
    val_split.set_transforms()
    if val_split.extract_features:
        print("WARNING: extracting validation set features")
        if val_split.augment_features:
            print("WARNING: augmenting validation set features")
    if test_split.extract_features:
        print("WARNING: extracting test set features")
        if test_split.augment_features:
            print("WARNING: augmenting test set features")
    val_split.set_transforms()
    test_split.set_transforms()
        
    workers = 6
    if args.debug_loader:
        workers = 1
    train_loader = get_split_loader(train_split, training=True, weighted = args.weighted_sample, workers=workers)
    val_loader = get_split_loader(val_split,  workers=workers)
    test_loader = get_split_loader(test_split, workers=workers)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping and not args.tuning:
        early_stopping = EarlyStopping(min_epochs = args.min_epochs, patience = 50, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    use_clam = False
    if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
        use_clam = True

    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=args.lr_factor, patience=args.lr_patience, verbose = True)
    
    for epoch in range(args.max_epochs):
        ## train a loop and evaluate validation set
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, feature_extractor=feature_extractor_model, debug_loader=args.debug_loader, clam=use_clam)
        stop, val_acc, _, _, val_auc, val_loss, _, _ = evaluate(model, val_loader, args.n_classes, "validate", cur, epoch, early_stopping, writer, loss_fn_val, args.results_dir,feature_extractor=feature_extractor_model,clam=use_clam)
        
        scheduler.step(val_loss)   
        if args.tuning:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=val_loss, accuracy=val_acc, auc=val_auc)

        if stop: 
            break
     
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_acc, val_bal_acc, val_f1, val_auc, _, _, _= evaluate(model, val_loader, args.n_classes, "final")
    print('Final val acc: {:.4f}, bal acc: {:.4f}, f1: {:.4f}, ROC AUC: {:.4f}'.format(val_acc, val_bal_acc, val_f1, val_auc))

    _, test_acc, test_bal_acc, test_f1, test_auc, _, _, _ = evaluate(model, test_loader, args.n_classes, "final")
    print('Final test acc: {:.4f}, bal acc: {:.4f}, f1: {:.4f}, ROC AUC: {:.4f}'.format(test_acc, test_bal_acc, test_f1, test_auc))


    if args.tuning:
        output_row=[args.reg,args.lr,args.drop_out,val_loss,val_auc,val_acc]
        output_dataframe=pd.read_csv("/CLAM/"+args.tuning_output_file)
        output_dataframe.loc[len(output_dataframe)] = output_row
        output_dataframe.to_csv("/CLAM/"+args.tuning_output_file,index=False)

    if writer:
        writer.add_scalar('final/val_accuracy', val_acc, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_accuracy', test_acc, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return test_auc, val_auc, test_acc, val_acc 


def train_loop(epoch, model, loader, optimizer, n_classes, bag_weight=0.5, writer = None, loss_fn = None, feature_extractor = None, debug_loader = False, clam = False):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    if feature_extractor is not None:
        feature_extractor.eval()
    train_loss = 0.

    if clam:
        inst_logger = Accuracy_Logger(n_classes=n_classes)
        train_inst_loss = 0.
        inst_count = 0
    
    all_probs = np.zeros((len(loader), n_classes))
    all_preds = np.zeros(len(loader))
    all_labels = np.zeros(len(loader))

    print('\n')
    for batch_idx, inputs in enumerate(loader):
        if len(inputs)==2:
            data,label = inputs
        else:
            data,adj,label = inputs
            adj = adj.to(device,non_blocking=True)

        if debug_loader:
            continue
        
        plot_data=False ##plot_data is not yet callable
        if plot_data:
            pil_image_transform=transforms.ToPILImage()
            for patch in data:
                plot_tensor = patch
                plot_image=pil_image_transform(plot_tensor)
                plot_image.save("../mount_outputs/patch_plots_hipt/{}.jpg".format(random.randint(0,100000)))
        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
        
        if feature_extractor is not None:
            with torch.no_grad():
                data = feature_extractor(data)
        
        model.train()
        if len(inputs)==3:
            logits, Y_prob, Y_hat, _, _ = model(data, adj, training=True) ##dont need clam options here as they don't work for graph models
        else:
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=clam)
        
        bag_loss = loss_fn(logits, label)

        if clam:
            instance_loss = instance_dict['instance_loss']
            inst_count+=1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value
            loss = bag_weight * bag_loss + (1-bag_weight) * instance_loss
            
            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

        else:
            loss = bag_loss

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        probs = Y_prob.detach().cpu().numpy()
        all_probs[batch_idx] = probs
        all_preds[batch_idx] = Y_hat
        all_labels[batch_idx] = label

        #if (batch_idx + 1) % 1000 == 0:
        #    print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))

    if math.isnan(train_loss):
        assert 1==2,"nan training loss"

    # calculate loss
    train_loss /= len(loader)

    print("Training")
    accuracy, balanced_accuracy, f1, auc = compute_metrics(all_probs, all_preds, all_labels, n_classes)
    print('Epoch: {}, train_loss: {:.4f}, acc: {:.4f}, bal_acc: {:.4f}, f1: {:.4f}, auc: {:.4f}'.format(epoch,loss, accuracy, balanced_accuracy, f1, auc))

    if clam:
        if inst_count > 0:
            train_inst_loss /= inst_count
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', accuracy, epoch)
        writer.add_scalar('train/bal_accuracy', balanced_accuracy, epoch)
        writer.add_scalar('train/f1', f1, epoch)
        writer.add_scalar('train/auc', auc, epoch)
        if clam:
            writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def compute_metrics(probs,preds,labels,n_classes):
    accuracy = accuracy_score(labels,preds)
    balanced_accuracy = balanced_accuracy_score(labels,preds)
    
    if n_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
        f1 = f1_score(labels,preds)
    else:
        try:
            auc = roc_auc_score(labels, probs,multi_class='ovr')
        except:
            auc = -1.
        f1 = f1_score(labels,preds,average='macro')
    
    labels = labels.tolist()
    preds = preds.tolist()
    for label_class in range(n_classes):
        count_label = labels.count(label_class)
        count_correct = sum(x == label_class and y == label_class for x, y in zip(preds,labels))
        if count_label > 0:
            acc = float(count_correct) / count_label
            print('class {}: acc {}, correct {}/{}'.format(label_class, acc, count_correct, count_label))

    return accuracy, balanced_accuracy, f1, auc

def evaluate(model, loader, n_classes, mode,cur=None,epoch=None,early_stopping = None, writer = None, loss_fn = None, results_dir=None, feature_extractor = None, clam=False):
    assert mode in ["validate","final"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss = 0.
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if clam:
        inst_logger = Accuracy_Logger(n_classes=n_classes)
        inst_loss = 0.
        inst_count=0

    all_probs = np.zeros((len(loader), n_classes))
    all_preds = np.zeros(len(loader))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']

    for batch_idx, inputs in enumerate(loader):
        if len(inputs)==2:
            data,label = inputs
        else:
            data,adj,label = inputs
            adj = adj.to(device,non_blocking=True)

        data, label = data.to(device,non_blocking=True), label.to(device,non_blocking=True)
        with torch.no_grad():
            if len(inputs)==3:
                logits, Y_prob, Y_hat, _, _ = model(data, adj, training=False)
            else:
                if clam:
                    logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
                else:
                    logits, Y_prob, Y_hat, _, _ = model(data)
        
        loss_value = loss_fn(logits, label)
        loss += loss_value.item()
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_preds[batch_idx] = Y_hat
        all_labels[batch_idx] = label

        if mode=="validate":
            if clam:
                instance_loss = instance_dict['instance_loss']
                inst_count += 1
                inst_loss += instance_loss.item()
                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)

    
    if math.isnan(loss):
        assert 1==2,"nan evaluation loss"

    loss /= len(loader)
    
    if mode == "validate":
        print("\nValidation")
    else:
        print("\nFinal Evaluation")
    accuracy, balanced_accuracy, f1, auc = compute_metrics(all_probs,all_preds,all_labels,n_classes)
    
    if clam:
        if inst_count > 0:
            inst_loss /= inst_count
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if mode == "validate":
        print('Val Set, val_loss: {:.4f}, acc: {:.4f}, bal_acc: {:.4f}, f1: {:.4f}, auc: {:.4f}'.format(loss, accuracy, balanced_accuracy, f1, auc))
        if writer:
            writer.add_scalar('val/loss', loss, epoch)
            writer.add_scalar('val/accuracy', accuracy, epoch)
            writer.add_scalar('val/bal_accuracy', balanced_accuracy, epoch)
            writer.add_scalar('val/f1', f1, epoch)
            writer.add_scalar('val/auc', auc, epoch)
            if clam:
                 writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

        if early_stopping:
            assert results_dir
            early_stopping(epoch, loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
            if early_stopping.early_stop:
                with open(os.path.join(results_dir,'early_stopping{}.txt'.format(cur)), 'w') as f:
                    f.write('Finished at epoch {}'.format(epoch))
                print("Early stopping")
                return True, accuracy, balanced_accuracy, f1, auc, loss, _, None
    
    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return False, accuracy, balanced_accuracy, f1, auc, loss, _, df
