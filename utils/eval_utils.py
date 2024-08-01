import numpy as np
#import cupy as np
import torch
import torch.nn as nn
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_graph import Graph_Model
from models.model_graph_mil import PatchGCN
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger, evaluate, seed_torch
from utils.sampling_utils import generate_sample_idxs, generate_features_array, update_sampling_weights, plot_sampling, plot_sampling_gif, plot_weighting, plot_weighting_gif
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import random
from sklearn.neighbors import NearestNeighbors
import openslide

from datasets.dataset_h5 import Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
from datasets.dataset_generic import Generic_MIL_Dataset

from ray import tune


def initiate_model(args, ckpt_path, num_features=0):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'encoding_size': args.encoding_size}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type in ['graph','graph_ms']:
         model = Graph_Model(pooling_factor=args.pooling_factor, pooling_layers=args.pooling_layers,  message_passings=args.message_passings, embedding_size=args.embedding_size,num_features=num_features, num_classes=args.n_classes,drop_out=args.drop_out, message_passing=args.message_passing, pooling=args.pooling)
    elif args.model_type =='patchgcn':
         model_dict = {'num_layers': 4, 'edge_agg': 'spatial', 'resample': 0.00, 'n_classes': args.n_classes, 'dropout': args.drop_out, 'hidden_dim': args.embedding_size }
         model = PatchGCN(**model_dict)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)
    
    print("args.cpu_only disabled as it caused problems making heatmaps/blockmaps")
    #if args.cpu_only:
    #    ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))
    #else:
    #if args.cpu_only:
    #    ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))
    #else:
    ckpt=torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=False)
    model.relocate()
    model.eval()
    return model


def extract_features(args,loader,feature_extraction_model,use_cpu):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_cpu:
        device=torch.device("cpu")
    for count, (batch,coords) in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            features = feature_extraction_model(batch)
        if use_cpu:
            features=features.cpu()
        if count==0:
            all_features=features
        else:
            all_features=torch.cat((all_features,features))
    if use_cpu:
        all_features=all_features.to(device)
    return all_features


def eval(config, dataset, args, ckpt_path, class_counts = None):
    num_features = 0
    if len(dataset[0])==3:
        num_features = dataset[0][0].shape[1]
    model = initiate_model(args, ckpt_path, num_features)
    print("model on device:",next(model.parameters()).device)
    print('Init Loaders')
    
    if args.tuning:
        args.weight_smoothing=config["weight_smoothing"]
        args.resampling_iterations=config["resampling_iterations"]
        args.samples_per_iteration=int(640/(config["resampling_iterations"]))
        args.sampling_neighbors=config["sampling_neighbors"]
        args.sampling_random=config["sampling_random"]
        args.sampling_random_delta=config["sampling_random_delta"]
    
    if args.bag_loss == 'balanced_ce':
        ce_weights=[(1/class_counts[i])*(sum(class_counts)/len(class_counts)) for i in range(len(class_counts))]
        print("weighting cross entropy with weights {}".format(ce_weights))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights).to(device,non_blocking=True)).to(device,non_blocking=True)
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    if args.sampling:
        assert 0<=args.sampling_random<=1,"sampling_random needs to be between 0 and 1"
        dataset.load_from_h5(True)
        seed_torch(args.seed)
        test_error, auc, df, _, loss = evaluate_sampling(model,dataset, args, loss_fn=loss_fn)
        ## come back and get these working with evaluate_sampling
        f1 = None
        bal_acc = None
    else:
        loader = get_simple_loader(dataset, model_type=args.model_type)
        _, acc, bal_acc, f1, auc, loss, _, df = evaluate(model, loader, args.n_classes, "final", loss_fn=loss_fn)
        test_error = 1-acc

    if args.tuning:
        tune.report(accuracy=1-test_error, auc=auc)    
    print('test_error: ', test_error)
    print('auc: ', auc)
    return test_error, df, loss, f1, auc, bal_acc


def select_best_samples(num_best, sample_idxs, attn_scores, previous_idxs = [], previous_attns = []):
    ## this function only runs if not retaining all samples from previous sampling steps
    all_attns = attn_scores + previous_attns
    if len(all_attns) <= num_best:
        best_idxs = sample_idxs + previous_idxs
        best_attns = all_attns 
    else:
        all_idxs = sample_idxs + previous_idxs
        best_attn_idxs = [idx.item() for idx in np.argsort(all_attns)][::-1]
        best_idxs = [all_idxs[attn_idx] for attn_idx in best_attn_idxs][:num_best]
        best_attns = [all_attns[attn_idx] for attn_idx in best_attn_idxs][:num_best]
    return best_idxs, best_attns


def evaluate_sampling(model, dataset, args, loss_fn = None):
    assert args.sampling
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if args.tuning:
        same_slide_repeats=args.same_slide_repeats
    else:
        same_slide_repeats=1

    num_slides=len(dataset)*same_slide_repeats
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Collecting predictions per sampling iteration to view performance across resampling iterations
    Y_hats_per_sample = []
    probs_per_sample = []
    logits_per_sample = []
    labels_per_sample = []

    ## Collecting final predictions for overall analysis
    final_probs = []
    final_preds = np.zeros(num_slides)
    final_logits = []
    labels = np.zeros(num_slides)
    test_error = 0.
    loss = 0.
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)

    loader = get_simple_loader(dataset, model_type=args.model_type)
    slide_id_list=[]
    texture_dataset = []
        
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

    num_random=int(args.samples_per_iteration*args.sampling_random)
    total_samples_per_slide = (args.samples_per_iteration*args.resampling_iterations)+args.final_sample_size
    
    if args.fully_random:
        total_samples_per_slide=args.samples_per_iteration
    print("Total patches sampled per slide: ",total_samples_per_slide)
    
    for batch_idx, contents in enumerate(loader):
        if not args.tuning:
            print('progress: {}/{}'.format(batch_idx, num_slides))
        
        ## unpack loader and calculate nearest neighbors
        (data, label,coords,slide_id) = contents
        coords=torch.tensor(coords)
        X = generate_features_array(args, data, coords, slide_id, slide_id_list, texture_dataset)
        nbrs = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
        data, label, coords = data.to(device), label.to(device), coords.to(device)
        slide_id=slide_id[0][0]


        for repeat_no in range(same_slide_repeats):
            samples_per_iteration=args.samples_per_iteration
            sampling_weights=np.full(shape=len(coords),fill_value=0.0001)

            ## Using all samples when required
            if args.fully_random or total_samples_per_slide>=len(coords):
                if total_samples_per_slide>=len(coords): 
                    print("full slide used for slide {} with {} patches".format(slide_id,len(coords)))
                    data_sample=data
                else:
                    sample_idxs=generate_sample_idxs(len(coords),[],[],samples_per_iteration,num_random=samples_per_iteration,grid=args.initial_grid_sample,coords=coords)
                    data_sample=data[sample_idxs].to(device)
                    
                with torch.no_grad():
                    logits, Y_prob, Y_hat, raw_attention, _ = model(data_sample)
                probs = Y_prob.cpu().numpy()[0]
                ## no outputs per sample as this is the only sample
                
                ## Store final outputs
                final_probs.append(probs)
                final_preds[batch_idx] = Y_hat.item()
                final_logits.append(logits)
                labels[(batch_idx*same_slide_repeats)+repeat_no] = label.item()

                #calculate loss and error
                loss_value = loss_fn(logits, label)
                loss += loss_value.item()
                error = calculate_error(Y_hat, label)
                test_error += error
                continue

            ## INITIAL SAMPLING ITERATION
            ## get new sample
            sample_idxs=generate_sample_idxs(len(coords),[],[],samples_per_iteration,num_random=samples_per_iteration,grid=args.initial_grid_sample,coords=coords)
            all_sample_idxs=sample_idxs
            data_sample=data[sample_idxs].to(device)

            ## run classifier on new samples
            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)
            probs = Y_prob.cpu().numpy()[0]
            attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0]#.cpu()
            attn_scores_list=raw_attention[0].cpu().tolist()
            
            ## find best samples to keep if not keeping all previous samples
            if not args.use_all_samples:
                best_sample_idxs, best_attn_scores = select_best_samples(args.retain_best_samples, sample_idxs, attn_scores_list, previous_idxs = [], previous_attns = [])

            ## update gifs
            if args.plot_sampling_gif:
                slide=plot_sampling_gif(slide_id,coords[sample_idxs],args,0,slide=None,final_iteration=False)
            if args.plot_weighting_gif:
                slide,x_coords,y_coords=plot_weighting_gif(slide_id,coords[all_sample_idxs],coords,sampling_weights,args,0,slide=None,final_iteration=False)

            ## Store outputs per iteration
            Y_hats_per_sample.append(Y_hat)
            probs_per_sample.append(probs)
            logits_per_sample.append(logits)
            labels_per_sample.append(label)

            ## Find nearest neighbors of each patch to prepare for spatial resampling
            distances, indices = nbrs.kneighbors(X[sample_idxs])
        
            ## INTERMEDIATE SAMPLING ITERATIONS
            sampling_random=args.sampling_random
            neighbors=args.sampling_neighbors
            for iteration_count in range(args.resampling_iterations-1):
                if sampling_random>args.sampling_random_delta:
                    sampling_random=sampling_random-args.sampling_random_delta
                else:
                    sampling_random=0
                num_random=int(samples_per_iteration*sampling_random)
                                                                        
                ## get new sample
                sampling_weights = update_sampling_weights(sampling_weights, attention_scores, all_sample_idxs, indices, neighbors, power=args.weight_smoothing, normalise=False, sampling_update=args.sampling_update, repeats_allowed=False)
                sample_idxs = generate_sample_idxs(len(coords), all_sample_idxs, sampling_weights/sum(sampling_weights), samples_per_iteration, num_random)
                distances, indices = nbrs.kneighbors(X[sample_idxs])
                
                ## update gifs - may be possible to simplify this 
                if args.plot_weighting_gif:
                    plot_weighting_gif(slide_id, coords[all_sample_idxs], coords, sampling_weights, args, iteration_count+1, slide=slide, x_coords=x_coords, y_coords=y_coords, final_iteration=False)
                if args.plot_sampling_gif:
                    if args.use_all_samples:
                        plot_sampling_gif(slide_id,coords[all_sample_idxs+sample_idxs],args,iteration_count+1,slide,final_iteration=False)
                    else:
                        plot_sampling_gif(slide_id,coords[sample_idxs],args,iteration_count+1,slide,final_iteration=False)

                ## store new sample ids
                all_sample_idxs=all_sample_idxs+sample_idxs

                ## get new sample features
                data_sample=data[sample_idxs].to(device)

                ## run classifier on new samples
                with torch.no_grad():
                    logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)
                probs = Y_prob.cpu().numpy()[0]
                attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0].cpu()
                attention_scores=attention_scores[-samples_per_iteration:]
                attn_scores_list=raw_attention[0].cpu().tolist()

                ## find best samples to retain if not keeping all previous samples
                if not args.use_all_samples:
                    best_sample_idxs, best_attn_scores = select_best_samples(args.retain_best_samples, sample_idxs, attn_scores_list, previous_idxs = best_sample_idxs, previous_attns = best_attn_scores)
                                              
                ## Store outputs per iteration
                Y_hats_per_sample.append(Y_hat)
                probs_per_sample.append(probs)
                logits_per_sample.append(logits)
                labels_per_sample.append(label)

                ## update neighbors parameter
                neighbors=neighbors-args.sampling_neighbors_delta
        
            ## FINAL SAMPLING ITERATION
            ## get new sample
            sampling_weights=update_sampling_weights(sampling_weights,attention_scores,all_sample_idxs,indices,neighbors,power=args.weight_smoothing,normalise=False,sampling_update=args.sampling_update,repeats_allowed=False)
            if args.use_all_samples:
                sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),args.final_sample_size,num_random=0)
                sample_idxs=sample_idxs+all_sample_idxs
                all_sample_idxs=sample_idxs
            else:
                sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),int(args.final_sample_size-len(best_sample_idxs)),num_random=0)
                all_sample_idxs=all_sample_idxs+sample_idxs
                sample_idxs=sample_idxs+best_sample_idxs
    
            data_sample=data[sample_idxs].to(device)
            ## run classifier on new samples
            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)
            probs = Y_prob.cpu().numpy()[0]

            acc_logger.log(Y_hat, label)
            
            ## Store outputs per iteration
            Y_hats_per_sample.append(Y_hat)
            probs_per_sample.append(probs)
            logits_per_sample.append(logits)
            labels_per_sample.append(label)

            ## Store final outputs
            final_probs.append(probs)
            final_preds[batch_idx] = Y_hat.item()
            final_logits.append(logits)
            labels[(batch_idx*same_slide_repeats)+repeat_no] = label.item()

            ## Calculate loss and error using final sample
            loss_value = loss_fn(logits, label)
            loss += loss_value.item()
            error = calculate_error(Y_hat, label)
            test_error += error

            ## Plot anything wanting plotting
            if args.plot_sampling:
                plot_sampling(slide_id,coords[sample_idxs],args,Y_hat==label)
            if args.plot_sampling_gif:
                plot_sampling_gif(slide_id,coords[sample_idxs],args,iteration_count+1,Y_hat==label,slide,final_iteration=True)
            if args.plot_weighting:
                plot_weighting(slide_id,coords[all_sample_idxs],coords,sampling_weights,args,Y_hat==label)
            if args.plot_weighting_gif:
                plot_weighting_gif(slide_id, coords[all_sample_idxs], coords, sampling_weights, args, iteration_count+1, slide=slide, x_coords=x_coords, y_coords=y_coords, final_iteration=True)

    
    all_errors=[]
    probs_per_sample = np.array(probs_per_sample)
    try:
        for i in range(args.resampling_iterations+1):
            all_errors.append(round(calculate_error(torch.Tensor(Y_hats_per_sample[i::args.resampling_iterations+1]),torch.Tensor(labels_per_sample[i::args.resampling_iterations+1])),3))
        print("Error per sampling iteration: ",all_errors)
    except:
        print("error per iteration didn't run, likely caused by a slide being too small for sampling")
    
    all_aucs=[]
    if len(np.unique(labels)) == 2:
        try:
            for i in range(args.resampling_iterations+1):
                auc_score = roc_auc_score(labels,[yprob.tolist()[1] for yprob in probs_per_sample[i::args.resampling_iterations+1]])
                all_aucs.append(round(auc_score,3))
            print("AUC per sampling iteration: ",all_aucs)
        except:
            print("auc per iteration didn't run, likely caused by a slide being too small for sampling")
    else:
        print("AUC scoring by iteration not implemented for multi-class classification yet")
        
    test_error /= num_slides
    aucs = []
    final_probs = np.array(final_probs)

    if len(np.unique(labels)) == 2:
        auc_score = roc_auc_score(labels, final_probs[:,1])
    else:
        aucs = []
        n_classes=len(np.unique(labels))
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], final_probs[:,class_idx])
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

            auc_score = np.nanmean(np.array(aucs))

    results_dict = {'Y': labels, 'Y_hat': final_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): final_probs[:,c]})

    df = pd.DataFrame(results_dict)
    loss /= len(loader)
    return test_error, auc_score, df, acc_logger, loss

