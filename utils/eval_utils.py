import numpy as np
#import cupy as np
import torch
import torch.nn as nn
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
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


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
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
    model.load_state_dict(ckpt_clean, strict=True)
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


def eval(config, dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    print("model on device:",next(model.parameters()).device)
    print('Init Loaders')
    
    if args.tuning:
        args.weight_smoothing=config["weight_smoothing"]
        args.resampling_iterations=config["resampling_iterations"]
        args.samples_per_iteration=int(640/(config["resampling_iterations"]))
        args.sampling_neighbors=config["sampling_neighbors"]
        args.sampling_random=config["sampling_random"]
        args.sampling_random_delta=config["sampling_random_delta"]
    
    
    if args.eval_features:
        test_error, auc, df, _ = summary_sampling(model,dataset,args)

    elif args.sampling:
        assert 0<=args.sampling_random<=1,"sampling_random needs to be between 0 and 1"
        dataset.load_from_h5(True)
        #loader = get_simple_loader(dataset)
        test_error, auc, df, _ = summary_sampling(model,dataset, args)
    else:
        loader = get_simple_loader(dataset)
        test_error, auc, df, _, loss = summary(model, loader, args)
    
    if args.tuning:
        tune.report(accuracy=1-test_error, auc=auc)    
    print('test_error: ', test_error)
    print('auc: ', auc)
    return test_error, auc, df, loss


def summary(model, loader, args, loss_fn = None):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    loss_value = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        loss = loss_fn(logits, label)
        loss_value += loss.item()
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)
    loss_value /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return test_error, auc_score, df, acc_logger, loss_value


def summary_sampling(model, dataset, args):
    model.eval()
    
    if args.tuning:
        same_slide_repeats=args.same_slide_repeats
    else:
        same_slide_repeats=1

    num_slides=len(dataset)*same_slide_repeats
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if args.cpu_only:
    #    device=torch.device("cpu")
    test_loss = 0.
    test_error = 0.

    if args.sampling_average:
        sampling_update='average'
    else:
        sampling_update='max'

    ## Collecting Y_hats and labels to view performance across resampling iterations
    Y_hats=[]
    labels=[]
    Y_probs=[]
    all_logits=[]
    all_probs=[]
    all_labels_byrep=[]
    if args.eval_features:
        if args.cpu_only:
            device=torch.device("cpu")

        kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {} 
        feature_extraction_model=resnet50_baseline(pretrained=True,dataset=args.pretraining_dataset)
        feature_extraction_model = feature_extraction_model.to(device)
        
        label_dict=args.label_dict
        all_labels = pd.read_csv(args.csv_path)['label']
        all_labels=[label_dict[key] for key in all_labels]
        all_labels_tensor=torch.Tensor(all_labels)
        iterator=range(len(dataset))
    
    else:
        loader = get_simple_loader(dataset)
        all_labels = np.zeros(num_slides)
        slide_ids = loader.dataset.slide_data['slide_id']
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
        iterator=loader

    acc_logger = Accuracy_Logger(n_classes=args.n_classes)

    test_loss = 0.
    test_error = 0.
    #all_probs = np.zeros((num_slides, args.n_classes))
    all_preds = np.zeros(num_slides)
    
    num_random=int(args.samples_per_iteration*args.sampling_random)
    
    if args.fully_random:
        total_samples_per_slide=args.samples_per_iteration    
    else:
        total_samples_per_slide = (args.samples_per_iteration*args.resampling_iterations)+args.final_sample_size
    if args.sampling:
        print("Total patches sampled per slide: ",total_samples_per_slide)
    
    for batch_idx, contents in enumerate(iterator):
        if not args.tuning and not args.fully_random:
            print('\nprogress: {}/{}'.format(batch_idx, num_slides))
        
        ## get features, either by calculating now or loading from file
        if args.eval_features:
            label=all_labels[batch_idx]
            label_tensor=all_labels_tensor[batch_idx]
            if isinstance(dataset[batch_idx],np.int64):
                slide_id=str(dataset[batch_idx])
            else:
                slide_id = dataset[batch_idx].split(args.slide_ext)[0]

            bag_name = slide_id+'.h5'
            h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
            slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)

            wsi = openslide.open_slide(slide_file_path)
            sampled_data = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, pretrained=True,
                        custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
            coords=sampled_data.coords(len(sampled_data))
            X = np.array(coords)

        else:
            (data, label,coords,slide_id) = contents
            coords=torch.tensor(coords)
            X = generate_features_array(args, data, coords, slide_id, slide_id_list, texture_dataset)
            nbrs = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
            data, label, coords = data.to(device), label.to(device), coords.to(device)
            slide_id=slide_id[0][0]


        for repeat_no in range(same_slide_repeats):
            samples_per_iteration=args.samples_per_iteration
            ## Generate initial sample_idsx
            #print("available patches:", len(coords))
            if not args.sampling or args.fully_random or total_samples_per_slide>=len(coords):
                if not args.sampling or total_samples_per_slide>=len(coords): 
                    print("full slide used for slide {} with {} patches".format(slide_id,len(coords)))
                    if args.eval_features:
                        sample_idxs=generate_sample_idxs(len(coords),[],[],len(coords),num_random=len(coords),grid=False,coords=coords)
                        sampled_data.update_sample(sample_idxs)
                        loader = DataLoader(dataset=sampled_data, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)
                        data_sample=extract_features(args,loader,feature_extraction_model,use_cpu=args.cpu_only)
                        data_sample.to(device)
                    else:
                        data_sample=data
                else:
                    sample_idxs=generate_sample_idxs(len(coords),[],[],samples_per_iteration,num_random=samples_per_iteration,grid=args.initial_grid_sample,coords=coords)
                    if args.eval_features:
                        sampled_data.update_sample(sample_idxs)
                        loader = DataLoader(dataset=sampled_data, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)
                        data_sample.to(device)
                    else:
                        data_sample=data[sample_idxs].to(device)
                    
                with torch.no_grad():
                    logits, Y_prob, Y_hat, raw_attention, _ = model(data_sample)
                Y_hats.append(Y_hat)
                acc_logger.log(Y_hat, label)
                probs = Y_prob.cpu().numpy()
                 
                all_probs.append(probs[0])
                

                if args.plot_weighting:
                    attention_scores=raw_attention[0]
                    #attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0]
                    new_attentions = np.repeat(min(attention_scores.cpu()),len(coords))
                    for i in range(len(sample_idxs)):
                        new_attentions[sample_idxs[i]]=attention_scores[i]
                        #new_attentions[sample_idxs[i]]=pow(attention_scores[i],args.weight_smoothing)
                    plot_weighting(slide_id,coords,new_attentions,args,Y_hat==label)
                
                if args.eval_features:
                    all_labels_byrep.append(label)
                else:
                    all_labels_byrep.append(label[0].item())
                #all_probs[(batch_idx*same_slide_repeats)+repeat_no] = probs
                all_preds[(batch_idx*same_slide_repeats)+repeat_no] = Y_hat.item()
                if args.eval_features:
                    error = calculate_error(Y_hat, label_tensor)
                else:
                    all_labels[(batch_idx*same_slide_repeats)+repeat_no] = label.item()
                    error = calculate_error(Y_hat, label)
                test_error += error
                continue

            ## Inital sample
            sample_idxs=generate_sample_idxs(len(coords),[],[],samples_per_iteration,num_random=samples_per_iteration,grid=args.initial_grid_sample,coords=coords)
            all_sample_idxs=sample_idxs
            sampling_weights=np.full(shape=len(coords),fill_value=0.0001)
            if args.eval_features:
                sampled_data.update_sample(sample_idxs)
                loader = DataLoader(dataset=sampled_data, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)
                if len(sample_idxs)>20000:
                    print("very large slide - using cpu. Disable this functionality if using beefy GPU (approx 8GB GPU ram or more)")
                    data_sample=extract_features(args,loader,feature_extraction_model,use_cpu=True)
                else:
                    data_sample=extract_features(args,loader,feature_extraction_model,use_cpu=False)
                data_sample.to(device)
                all_previous_features=data_sample
                all_sample_idxs=sample_idxs
            else:
                data_sample=data[sample_idxs].to(device)

            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)
            attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0]#.cpu()
            attn_scores_list=raw_attention[0].cpu().tolist()
        
            if not args.use_all_samples:
                if args.samples_per_iteration<=args.retain_best_samples:
                    best_sample_idxs=sample_idxs
                    best_attn_scores=attn_scores_list
                else:
                    attn_idxs=[idx.item() for idx in np.argsort(attn_scores_list)][::-1]
                    best_sample_idxs=[sample_idxs[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
                    best_attn_scores=[attn_scores_list[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]

            if args.plot_sampling_gif:
                slide=plot_sampling_gif(slide_id,coords[sample_idxs],args,0,slide=None,final_iteration=False)
        
            if args.plot_weighting_gif:
                slide,x_coords,y_coords=plot_weighting_gif(slide_id,coords[all_sample_idxs],coords,sampling_weights,args,0,slide=None,final_iteration=False)

            Y_hats.append(Y_hat)
            labels.append(label)
            Y_probs.append(Y_prob)
            all_logits.append(logits)

            ## Find nearest neighbors of each patch to prepare for spatial resampling
            nbrs = NearestNeighbors(n_neighbors=args.sampling_neighbors, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X[sample_idxs])
        
            ##Subsequent iterations
            sampling_random=args.sampling_random
            neighbors=args.sampling_neighbors
            for iteration_count in range(args.resampling_iterations-1):
                if sampling_random>args.sampling_random_delta:
                    sampling_random=sampling_random-args.sampling_random_delta
                else:
                    sampling_random=0
                num_random=int(samples_per_iteration*sampling_random)
                #attention_scores=attention_scores/max(attention_scores)
                                                                        
                sampling_weights=update_sampling_weights(sampling_weights,attention_scores,all_sample_idxs,indices,neighbors,power=args.weight_smoothing,normalise=False,
                                        sampling_update=sampling_update,repeats_allowed=False)
                #sampling_weights_over20=[weight for weight in sampling_weights if weight>0.2]
                #print(sampling_weights)
                #print(max(sampling_weights))
                #print(len(sampling_weights_over20))
                if args.plot_weighting_gif:
                                plot_weighting_gif(slide_id,coords[all_sample_idxs],coords,sampling_weights,args,iteration_count+1,slide,x_coords,y_coords,final_iteration=False)
                sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),samples_per_iteration,num_random)
                distances, indices = nbrs.kneighbors(X[sample_idxs])
                all_sample_idxs=all_sample_idxs+sample_idxs
                
                if args.plot_sampling_gif:
                    if args.use_all_samples:
                        plot_sampling_gif(slide_id,coords[all_sample_idxs],args,iteration_count+1,slide,final_iteration=False)
                    else:
                        plot_sampling_gif(slide_id,coords[sample_idxs],args,iteration_count+1,slide,final_iteration=False)

                if args.eval_features:
                    sampled_data.update_sample(sample_idxs)
                    loader = DataLoader(dataset=sampled_data, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)
                    data_sample=extract_features(args,loader,feature_extraction_model,use_cpu=False)
                    all_previous_features=torch.cat((all_previous_features,data_sample))
                    if args.use_all_samples:
                        if iteration_count==args.resampling_iterations-2: 
                            data_sample=all_previous_features
                    data_sample.to(device)
                else:
                    data_sample=data[sample_idxs].to(device)


                with torch.no_grad():
                    logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)
                attention_scores=torch.nn.functional.softmax(raw_attention,dim=1)[0].cpu()
                attention_scores=attention_scores[-samples_per_iteration:]
                attn_scores_list=raw_attention[0].cpu().tolist()

                if not args.use_all_samples:
                    attn_scores_combined=attn_scores_list+best_attn_scores
                    idxs_combined=sample_idxs+best_sample_idxs

                    if len(idxs_combined)<=args.retain_best_samples:
                        best_sample_idxs=idxs_combined
                        best_attn_scores=attn_scores_combined
                    else:
                        attn_idxs=[idx.item() for idx in np.argsort(attn_scores_combined)][::-1]
                        best_sample_idxs=[idxs_combined[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
                        best_attn_scores=[attn_scores_combined[attn_idx] for attn_idx in attn_idxs][:args.retain_best_samples]
                Y_hats.append(Y_hat)
                labels.append(label)
                Y_probs.append(Y_prob)
                all_logits.append(logits)
                                                             
                neighbors=neighbors-args.sampling_neighbors_delta
        
            ## Final sampling iteration
            sampling_weights=update_sampling_weights(sampling_weights,attention_scores,all_sample_idxs,indices,neighbors,power=args.weight_smoothing,normalise=False,
                                sampling_update=sampling_update,repeats_allowed=False)
            if args.use_all_samples:
                sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),args.final_sample_size,num_random=0)
                sample_idxs=sample_idxs+all_sample_idxs
                all_sample_idxs=sample_idxs
            else:
                sample_idxs=generate_sample_idxs(len(coords),all_sample_idxs,sampling_weights/sum(sampling_weights),int(args.final_sample_size-len(best_sample_idxs)),num_random=0)
                all_sample_idxs=all_sample_idxs+sample_idxs
                sample_idxs=sample_idxs+best_sample_idxs
    
            if args.eval_features:
                sampled_data.update_sample(sample_idxs)
                loader = DataLoader(dataset=sampled_data, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)
                data_sample=extract_features(args,loader,feature_extraction_model,use_cpu=args.cpu_only)
                all_previous_features=torch.cat((all_previous_features,data_sample))
                if args.use_all_samples:
                    data_sample=all_previous_features
                data_sample.to(device)
            else:
                data_sample=data[sample_idxs].to(device)

            with torch.no_grad():
                logits, Y_prob, Y_hat, raw_attention, results_dict = model(data_sample)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            
            #all_probs[(batch_idx*same_slide_repeats)+repeat_no] = probs
            all_probs.append(probs[0])
            all_labels_byrep.append(label[0].item())
            all_preds[(batch_idx*same_slide_repeats)+repeat_no] = Y_hat.item()
            
            if args.plot_sampling:
                plot_sampling(slide_id,coords[sample_idxs],args,Y_hat==label)
            if args.plot_sampling_gif:
                plot_sampling_gif(slide_id,coords[sample_idxs],args,iteration_count+1,Y_hat==label,slide,final_iteration=True)
            if args.plot_weighting:
                plot_weighting(slide_id,coords,sampling_weights,args,Y_hat==label)
            if args.plot_weighting_gif:
                plot_weighting_gif(slide_id,coords[all_sample_idxs],coords,sampling_weights,args,iteration_count+1,Y_hat==label,slide,x_coords,y_coords,final_iteration=True)


            if args.eval_features:
                error = calculate_error(Y_hat, label_tensor)
            else:
                all_labels[(batch_idx*same_slide_repeats)+repeat_no] = label.item()
                error = calculate_error(Y_hat, label)

            test_error += error
            #print("len all sample idxs",len(all_sample_idxs))
    
    #assert 1==2, "probs {} labels {}".format(len(all_probs),len(all_labels))
    all_errors=[]
    #if args.eval_features:
      #  #for i in range(args.resampling_iterations):
            #all_errors.append(round(calculate_error(torch.Tensor(Y_hats[i::args.resampling_iterations]),torch.Tensor(all_labels)),3))
    if not args.eval_features:
        for i in range(args.resampling_iterations):
            all_errors.append(round(calculate_error(torch.Tensor(Y_hats[i::args.resampling_iterations]),torch.Tensor(labels[i::args.resampling_iterations])),3))

    
        all_aucs=[]
        if len(np.unique(all_labels)) == 2:
            if len(all_labels)==len([yprob.tolist()[0][1] for yprob in Y_probs[0::args.resampling_iterations]]):
                for i in range(args.resampling_iterations):
                    auc_score = roc_auc_score(all_labels,[yprob.tolist()[0][1] for yprob in Y_probs[i::args.resampling_iterations]])
                    all_aucs.append(round(auc_score,3))
                print("all aucs: ",all_aucs)
            else:
                print("scoring by iteration unavailable as not all slides could be sampled")
        else:
            print("AUC scoring by iteration not implemented for multi-class classification yet")
        
    test_error /= num_slides
    aucs = []
    all_probs=np.array(all_probs)
    #print(all_labels_byrep)
    #print(all_probs)
    #print(roc_auc_score(all_labels_byrep,all_probs[:,1]))
    #print(len(all_labels_byrep))
    if len(np.unique(all_labels)) == 2:
        #print(all_labels_byrep)
        #print(all_probs)
        auc_score = roc_auc_score(all_labels_byrep, all_probs[:,1])
    else:
        aucs = []
        n_classes=len(np.unique(all_labels))
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

            auc_score = np.nanmean(np.array(aucs))

    results_dict = {'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})

    df = pd.DataFrame(results_dict)
    #print("all errors: ",all_errors)
    #print("all aucs: ",all_aucs)
    return test_error, auc_score, df, acc_logger

