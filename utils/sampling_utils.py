import numpy as np
import math
import openslide
import matplotlib.pyplot as plt
import glob
from PIL import Image
#from matplotlib import colors
from multiprocessing.pool import Pool
import random
import os 

def generate_sample_idxs(idxs_length,previous_samples,sampling_weights,samples_per_iteration,num_random,grid=False,coords=None):
    if grid:
        assert len(coords)>0
        x_coords=[x.item() for x,y in coords]
        y_coords=[y.item() for x,y in coords]
        min_x=min(x_coords)
        max_x=max(x_coords)
        min_y=min(y_coords)
        max_y=max(y_coords)
        
        num_of_splits=int(math.sqrt(samples_per_iteration))
        x_borders=np.linspace(min_x,max_x+0.00001,num_of_splits+1)
        y_borders=np.linspace(min_y,max_y+0.00001,num_of_splits+1)
        
        sample_idxs=[]
        coords_splits=[[] for _ in range((num_of_splits+1)*(num_of_splits+1))]
        for coord_idx, (x,y) in enumerate(coords):
            x_border_idx=np.where(x_borders==max(x_borders[x_borders<=x.item()]))[0][0]
            y_border_idx=np.where(y_borders==max(y_borders[y_borders<=y.item()]))[0][0]
            coords_splits[(num_of_splits+1)*x_border_idx+y_border_idx].append(coord_idx)
        for coords_in_split in coords_splits:
            if len(coords_in_split)>0:
                sample_idxs=sample_idxs+list(np.random.choice(coords_in_split, size=1,replace=False))
        if len(sample_idxs)<samples_per_iteration:
            sample_idxs=sample_idxs+list(np.random.choice(range(0,len(coords)), size=samples_per_iteration-len(sample_idxs),replace=False))

    else:
        available_idxs=set(range(idxs_length))
        nonrandom_idxs=[]
        random_idxs=[]
        if int(samples_per_iteration-num_random)>0:
            nonrandom_idxs=list(np.random.choice(range(idxs_length),p=sampling_weights,size=int(samples_per_iteration-num_random),replace=False))
            previous_samples=previous_samples+nonrandom_idxs
            available_idxs=available_idxs-set(previous_samples)
        if num_random>0:
            random_idxs=random.sample(list(available_idxs),num_random)
        sample_idxs=random_idxs+nonrandom_idxs
    return sample_idxs


def generate_features_array(args, data, coords, slide_id, slide_id_list, texture_dataset):
    if args.sampling_type=='spatial':
        X = np.array(coords)
    elif args.sampling_type=='textural':
        assert args.texture_model in ['resnet50','levit_128s'], 'incorrect texture model chosen'
        if args.texture_model=='resnet50':
            X = np.array(data)
        else:
            texture_index=slide_id_list.index(slide_id[0][0])
            levit_features=texture_dataset[texture_index][0]
            assert len(levit_features)==len(data),"features length mismatch"
            X = np.array(levit_features)
    return X


def update_sampling_weights(sampling_weights, attention_scores, all_sample_idxs, indices, neighbors, power=0.15, normalise = True, sampling_update = 'max', repeats_allowed = False):
    """
    Updates the sampling weights of all patches by looping through the most recent sample and adjusting all neighbors weights
    By default the weight of a patch is the maximum of its previous weight and the newly assigned weight, though can be changed to average or simply to the newest available
    power is a hyperparameter controlling how attention scores are smoothed as typically very close to 0 or 1
    if repeated_allowed = False then weights for previous samples are set to 0
    """
    assert sampling_update in ['max','newest','average','none']
    new_attentions = np.zeros(shape=len(sampling_weights))
    #new_attentions=dict(enumerate(new_attentions)) ## may be better to skip np zeros and use a loop
    if sampling_update=='average':
        for i in range(len(indices)):
            for index in indices[i][:neighbors]:
                if new_attentions[index]>0:
                    ## not a perfect method of averaging but want it to run quickly
                    new_attentions[index]=(new_attentions[index]+attention_scores[i])/2
                else:
                    new_attentions[index]=attention_scores[i]
        new_attentions=pow(new_attentions,power)

        for i in range(len(sampling_weights)):
            if new_attentions[i] > 0:
                ## default sampling weights are 0.0001 to prevent instability from sparsity
                if sampling_weights[i] > 0.0002:
                    sampling_weights[i] = (sampling_weights[i] + new_attentions[i])/2 
                else:
                    sampling_weights[i] = new_attentions[i]
        ##old version
        #for i in range(len(indices)):
        #    for index in indices[i][:neighbors]:
                ## the default value is 0.0001
        #        if sampling_weights[index]>0.0001:
        #            sampling_weights[index]=(sampling_weights[index]+pow(attention_scores[i],power))/2
        #        else:
        #            sampling_weights[index]=pow(attention_scores[i],power)
    elif sampling_update=='max':
        ## this chunk is an idea to avoid doing two loops by only looping on the indices, haven't yet managed to improve speed
        #indices=np.array(indices)
        #used_indices=np.unique(indices)
        #for used_index in used_indices:
        #    rows = np.where(indices==used_index)[0]
        #    if len(rows)<2:
        #        new_attentions[used_index]=attention_scores[rows[0]]
        #    else:
        #        new_attentions[used_index]=max(attention_scores[rows])
        


        #new_attentions_dict={}
        #for i in range(len(indices)):
        #    for index in indices[i][:neighbors]:
        #        if index in new_attentions_dict:
        #            if attention_scores[i]>new_attentions_dict[index]:
         #               new_attentions_dict[index]=attention_scores[i]
         #       else:
         #               new_attentions_dict[index]=attention_scores[i]

        ## this block is currently faster
       
        
        ## WORKING CODE but no longer fastest 
        ########################################################
        #print("WARNING: USING OLDER VERSION OF CODE IN SAMPLING_UTILS")
        for i in range(len(indices)):
            for index in indices[i][:neighbors]:
                if new_attentions[index]>0:
                    if attention_scores[i]>new_attentions[index]:
                        new_attentions[index]=attention_scores[i]
                else:
                    new_attentions[index]=attention_scores[i]
        #######################################################
        
        ## this needs indices as a dict
        #indices=dict(indices)
        
        ## New fastest code - it has quartered the runtime of update_sampling_weights but certainly isnt working with eval.py (seems to be working with main.py - check this)
        #####################################
        #indices_dict={}
        #for i,row in enumerate(indices):
        #    indices_dict[i]=row 
        #for key,values in indices_dict.items():
        #    for index in values:
        #        if new_attentions[index]>0:
        #            if attention_scores[i]>new_attentions[index]:
        #                new_attentions[index]=attention_scores[key]
        #        else:
        #                new_attentions[index]=attention_scores[key]
        ########################################

        #with Pool() as pool:
         #   for result in pool.map(task, range(10)):
        #        new_attention[result[index]]=
        #
                #sampling_weights[index]=max(sampling_weights[index],pow(attention_scores[i],power))
        #for key in new_attentions_dict:
        #    new_attentions_dict[key]=pow(new_attentions_dict[key],power)
        #print(new_attentions)
        #print(" ")
        #print(list(new_attentions.values()))
        #assert 1==2,"break"
        
        ## if new_attentions is a dict need the next line
        #new_attentions=np.array(list(new_attentions.values()),dtype=float)
        new_attentions=pow(new_attentions,power)
        #new_attentions=1 / (1 + np.exp(-new_attentions))
        #new_attentions=np.array(new_attentions)
        #for i in range(len(new_attentions)):
            #new_attentions[i]=pow(new_attentions[i],power)
        for i in range(len(sampling_weights)):
                if new_attentions[i]>sampling_weights[i]:
                    sampling_weights[i]=new_attentions[i]
                #sampling_weights[i]=max(sampling_weights[i],pow(new_attentions[i],power))
    elif sampling_update=='newest':
        for i in range(len(indices)):
            for index in indices[i][:neighbors]:
                new_attentions[index]=attention_scores[i]
        new_attentions=pow(new_attentions,power)
        for i in range(len(sampling_weights)):
                if new_attentions[i]>sampling_weights[i]:
                    sampling_weights[i]=new_attentions[i]

    if not repeats_allowed:
        for sample_idx in all_sample_idxs:
            sampling_weights[sample_idx]=0

    if normalise:
        sampling_weights=sampling_weights/sum(sampling_weights)

    return sampling_weights


def plot_sampling(slide_id,sample_coords,args,correct=False,thumbnail_size=1000):
    print("Plotting slide {} with {} samples".format(slide_id,len(sample_coords)))
    os.makedirs(args.plot_dir+'sampling_maps/', exist_ok=True)
    
    slide = openslide.open_slide(args.data_slide_dir+"/"+slide_id+".svs")
    img = slide.get_thumbnail((thumbnail_size,thumbnail_size))
    plt.figure()
    plt.imshow(img)
    x_values, y_values = sample_coords.T
    x_values=(x_values+128)*(thumbnail_size/max(slide.dimensions))
    y_values=(y_values+128)*(thumbnail_size/max(slide.dimensions))
    x_values=x_values.cpu()
    y_values=y_values.cpu()
    plt.scatter(x_values,y_values,s=6)
    plt.axis('off')
    if correct:
        correct_str="correct"
    else:
        correct_str="incorrect"
    plt.savefig(args.plot_dir+'sampling_maps/{}_{}.png'.format(slide_id,correct_str), dpi=300, pad_inches = 0, bbox_inches='tight')
    plt.close()
 

def plot_sampling_gif(slide_id,sample_coords,args,iteration,correct=False,slide=None,final_iteration=False,thumbnail_size=1000):
    if slide==None:
        slide = openslide.open_slide(args.data_slide_dir+"/"+slide_id+".svs")
        os.makedirs(args.plot_dir+'sampling_maps/gifs/stills/', exist_ok=True)

    img = slide.get_thumbnail((thumbnail_size,thumbnail_size))
    plt.figure()
    plt.imshow(img)
    x_values, y_values = sample_coords.T
    x_values=(x_values+128)*(thumbnail_size/max(slide.dimensions))
    y_values=(y_values+128)*(thumbnail_size/max(slide.dimensions))
    x_values=x_values.cpu()
    y_values=y_values.cpu()
    plt.scatter(x_values,y_values,s=6)
    plt.axis('off')
    plt.savefig(args.plot_dir+'sampling_maps/gifs/stills/{}_iter{}.png'.format(slide_id,str(iteration).zfill(3)), dpi=300,bbox_inches='tight',pad_inches = 0)
    plt.close()
    
    if final_iteration:
        print("Plotting gif for slide {} over {} iterations".format(slide_id,iteration+1))
        fp_in = args.plot_dir+"sampling_maps/gifs/stills/{}_iter*.png".format(slide_id)
        if correct:
            correct_str="correct"
        else:
            correct_str="incorrect"
        fp_out = args.plot_dir+"sampling_maps/gifs/{}_{}.gif".format(slide_id,correct_str)
        imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)  # extract first image from iterator
        img.save(fp=fp_out, format='GIF', append_images=imgs,save_all=True, duration=200, loop=1)

    return slide


def plot_weighting(slide_id,sample_coords,coords,weights,args,correct=False,thumbnail_size=3000):
    print("Plotting final weights for slide {}.".format(slide_id))
    os.makedirs(args.plot_dir+'weight_maps/', exist_ok=True)

    slide = openslide.open_slide(args.data_slide_dir+"/"+slide_id+".svs")
    img = slide.get_thumbnail((thumbnail_size,thumbnail_size))
    plt.figure()
    plt.imshow(img)
    x_values, y_values = coords.T
    x_values=(x_values+128)*(thumbnail_size/max(slide.dimensions))
    y_values=(y_values+128)*(thumbnail_size/max(slide.dimensions))
    x_values=x_values.cpu()
    y_values=y_values.cpu()
    
    plt.scatter(x_values, y_values, c=weights, cmap='jet', s=3, alpha=0.6, marker="s", edgecolors='none')
    x_samples, y_samples = sample_coords.T
    x_samples=(x_samples+128)*(thumbnail_size/max(slide.dimensions))
    y_samples=(y_samples+128)*(thumbnail_size/max(slide.dimensions))
    x_samples=x_samples.cpu()
    y_samples=y_samples.cpu()
    plt.scatter(x_samples,y_samples,c='white',s=3.5,alpha=1, edgecolors='none')

    plt.axis('off')
    if correct:
        correct_str="correct"
    else:
        correct_str="incorrect"
    plt.savefig(args.plot_dir+'weight_maps/{}_{}_{}.png'.format(slide_id,args.sampling_type,correct_str), dpi=500,pad_inches = 0, bbox_inches='tight')
    plt.close()


def plot_weighting_gif(slide_id,sample_coords,coords,weights,args,iteration,correct=False,slide=None,x_coords=None,y_coords=None,final_iteration=False,thumbnail_size=3000):
    if slide==None:
        slide = openslide.open_slide(args.data_slide_dir+"/"+slide_id+".svs")
        x_coords, y_coords = coords.T
        x_coords=(x_coords+128)*(thumbnail_size/max(slide.dimensions))
        y_coords=(y_coords+128)*(thumbnail_size/max(slide.dimensions))
        x_coords=x_coords.cpu()
        y_coords=y_coords.cpu()
        os.makedirs(args.plot_dir+'weight_maps/gifs/stills/', exist_ok=True)

    img = slide.get_thumbnail((thumbnail_size,thumbnail_size))
    plt.figure()
    plt.imshow(img)
    
    if iteration > 0:
        plt.scatter(x_coords, y_coords, c=weights, cmap='jet', s=3, alpha=0.6, marker="s", edgecolors='none')

        x_samples, y_samples = sample_coords.T
        x_samples=(x_samples+128)*(thumbnail_size/max(slide.dimensions))
        y_samples=(y_samples+128)*(thumbnail_size/max(slide.dimensions))
        x_samples=x_samples.cpu()
        y_samples=y_samples.cpu()
        plt.scatter(x_samples,y_samples,c='white',s=3.5,alpha=1, edgecolors='none')
        plt.axis('off')
        plt.savefig(args.plot_dir+'weight_maps/gifs/stills/{}_{}_iter{}.png'.format(slide_id,args.sampling_type,str(iteration).zfill(3)), dpi=300,bbox_inches='tight',pad_inches = 0)
        plt.close()
    
    if final_iteration:
        print("Plotting weight gif for slide {} over {} iterations".format(slide_id,iteration+1))
        fp_in = args.plot_dir+"weight_maps/gifs/stills/{}_{}_iter*.png".format(slide_id,args.sampling_type)
        if correct:
            correct_str="correct"
        else:
            correct_str="incorrect"
        fp_out = args.plot_dir+"weight_maps/gifs/{}_{}_{}.gif".format(slide_id,args.sampling_type,correct_str)
        imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)  # extract first image from iterator
        img.save(fp=fp_out, format='GIF', append_images=imgs,save_all=True, duration=500, loop=1)

    return slide, x_coords, y_coords

