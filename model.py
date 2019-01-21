# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 09:24:47 2019

@author: mrestrepo
"""
#%%
import os 
import csv 
from math import copysign 
import hashlib 
import json
#%%

import numpy as np 

import keras
from keras.models import Sequential 
from keras.layers import \
    Cropping2D, Lambda, Convolution2D, Dense, Flatten,  Dropout
    
from sklearn.model_selection import train_test_split
import sklearn

import cv2

#%%
assert(  keras.backend.image_data_format()  == 'channels_last' )
#%%
#TODO
# - add left/right images with perturbed angle...
# - add batch norm
# - clean up training frames, remove off-road...


# - IMPORTANT: For training we use cv2.imread  which yields GBR...
#    make sure simulator does the same...

#%%

INPUT_SHAPE = (160,320,3) # used at beginning of build_model

# We assume driving_log.csv and IMG directory are contained gere
DATA_DIR = r"C:/_DATA/autonomous-driving-nd/behavioral_cloning/"

if os.name != 'nt' : 
    DATA_DIR = "/home/workspace/"
#%%
def main() :
    #%%
    cfg = { "dropout_d1" : 0.6, # dropout rate at first dense layer
            "dropout_cs" : 0.4, # dropout rate at conv layers layer
            "batch_size" : 32, 
            "epochs" : 20, 
            "final_activation" : "tanh",
            "side_image_correction_mode" : "additive",
            "side_image_correction_param" : 0.10 }
    #%%
    samples = read_samples()
    #%%
    model_str = str( sorted( list( cfg.items() ))  )
    model_hash = hashlib.sha256( model_str.encode("utf-8") ).hexdigest()[:12] 
    
    with open( DATA_DIR + "model_" + model_hash + ".json", "wt") as f_out : 
        print( json.dumps(cfg), file=f_out)
    
    #%%
    model = train_model( cfg, samples )
    model.save( DATA_DIR + f"model_{model_hash}_{len(samples)}.h5" )
        
    #%%
    
def train_model( cfg, samples ) : 
    model = build_model( cfg )
        
    #angles= np.array( [ float( row[3] ) for row in samples] ) 
    #Angles are between -1 and 1.

    train_samples, non_train = train_test_split( samples, test_size = 0.3, random_state=42 )
    valid_samples, test_samples = train_test_split( non_train, test_size = 0.2, random_state=42 )
    
    #print( len(train_samples), len(valid_samples), len(test_samples))
    bs = cfg["batch_size"]
    
    train_generator = generator( train_samples, cfg  )
    valid_generator = generator( valid_samples, cfg )
        
    #%% Fit
    model.compile( loss='mse', optimizer='adam' )
    model.fit_generator( train_generator, 
                         steps_per_epoch=len(train_samples) // bs ,
                         validation_data=valid_generator,
                         validation_steps=len(valid_samples) // bs, 
                         epochs=cfg["epochs"])
    #%%
    
    return model


def build_model( cfg ) : 
    """Heavily inspired by code in Behavioral Cloning, 
       section 15. Even Mode Powerful Network
       Just added dropout"""
       
    model = Sequential() 
    model.add( Cropping2D(cropping=((70,25), (0,0)) , input_shape=INPUT_SHAPE ) )
    model.add( Lambda(lambda x : x / 255.0 - 0.5) ) 
    model.add( Convolution2D(24, (5,5), strides=(2,2), activation="relu") )
    model.add( Convolution2D(36, (5,5), strides=(2,2), activation="relu") )
    model.add( Dropout(cfg["dropout_cs"]) )
    model.add( Convolution2D(48, (5,5), strides=(2,2), activation="relu") )
    model.add( Dropout(cfg["dropout_cs"]) )
    model.add( Convolution2D(64, (3,3), activation="relu") )
    model.add( Dropout(cfg["dropout_cs"]) )    
    model.add( Convolution2D(64, (3,3), activation="relu") )
    model.add( Dropout(cfg["dropout_cs"]) )
    
    
    model.add( Flatten() )
    model.add( Dense(100, activation="relu") )
    model.add( Dropout(cfg["dropout_d1"]) )
    model.add( Dense(50, activation="relu") )
    model.add( Dense(1, activation=cfg["final_activation"]) )

    return model


def read_samples() :    
    
    with open( DATA_DIR + 'driving_log.csv' ) as csvfile : 
        reader = csv.reader( csvfile )
        samples = list( reader ) 
        
    print( f"samples from driving_log has {len(samples)}" )
    
    img_files = set( os.listdir( DATA_DIR + "IMG" ) ) 
    samples = [ row for row in samples if row[0].split("/")[-1] in img_files ]   
    
    print( f"samples that have img {len(samples)}" )
    
    return samples 


def generator_v0( samples, cfg ) :
    """Taken from section 18. Generators"""
    
    num_samples = len(samples)
    
    batch_size = cfg["batch_size"]
    
    step_size = batch_size // 2
    
    while True : 
        samples = sklearn.utils.shuffle(samples)
    
        for offset in range( 0, num_samples, step_size ) :
            batch_samples = samples[offset : offset + step_size ]
            images = []
            angles = []
            
            for sample in batch_samples: 
                fname = DATA_DIR + "IMG/" + sample[0].split('/')[-1]
                center_image = cv2.imread( fname )
                center_angle = float( sample[3] )
                images.append( center_image )
                angles.append( center_angle )
                
                mirrored_img = center_image[ : ,::-1, :] 
                images.append( mirrored_img )
                angles.append( -center_angle )
                
            X_train = np.array( images )
            y_train = np.array( angles )
            
            yield X_train, y_train


def generator( samples, cfg ) :
    """Taken from section 18. Generators"""
    
    num_samples = len(samples)
    
    batch_size = cfg["batch_size"]
    side_correct_mode  = cfg["side_image_correction_mode"]
    side_correct_param = cfg["side_image_correction_param"]
    
    if side_correct_mode == "additive" : 
        def correction_fn_left( angle )  :
            return angle + side_correct_param 
        def correction_fn_right( angle )  :
            return angle - side_correct_param 
        
    else : 
        def correct_fn_right( angle ) :
            # if angle > 0  :
            #    make it less positive
            #    return angle * ( 1 - side_correct_param)
            # elif angle < 0 : 
            #    make it more negative 
            #    return angle * ( 1 + side_correct_param)
            
            return angle * (1 - copysign( side_correct_param, angle ) )
        def correct_fn_left( angle ) :
            # if angle > 0  :
            #    make it more positive 
            #    return angle * ( 1 + side_correct_param)
            # elif angle < 0 : 
            #    make it less negative
            #    return angle * ( 1 - side_correct_param)
            return angle * ( 1 + copysign( side_correct_param, angle ))
        
    while True : 
        samples = sklearn.utils.shuffle(samples)
    
        for offset in range( 0, num_samples, batch_size ) :
            batch_samples = samples[offset : offset + batch_size ]
            img_angle_ps = []
            
            
            for sample in batch_samples:                 
                center_image = cv2.imread( get_img_fname( sample[0] )  )
                center_angle = float( sample[3] )
                img_angle_ps.append( (center_image, center_angle ) )
                                
                mirrored_img = center_image[ : ,::-1, :] 
                img_angle_ps.append( (mirrored_img, -center_angle ) )
                del center_image, mirrored_image
                
                if side_correct_mode  :                    
                    left_image  = cv2.imread( get_img_fname( sample[1] ) )                                        
                    left_angle = correction_fn_left( center_angle )
                    img_angle_ps.append( (left_image, left_angle ) )
                    left_mirrored = left_image[ :, ::-1, :]
                    img_angle_ps.append( (left_mirrored, -left_angle ) )
                                        
                    del left_image, left_angle, left_mirrored
                    
                    right_image  = cv2.imread( get_img_fname( sample[2] ) )                                        
                    right_angle = correction_fn_right( center_angle )
                    img_angle_ps.append( (right_image, right_angle ) )
                    right_mirrored = right_image[ :, ::-1, :]
                    img_angle_ps.append( (right_mirrored, -right_angle ) )
                                    
            X_train = np.array( [p[0] for p in img_angle_ps] )
            y_train = np.array( [p[1] for p in img_angle_ps] )
            
            yield X_train, y_train
            
        print( f"generator going around {num_samples} (last_offset={offset})")
                               
def get_img_fname( full_path ) :
    return DATA_DIR + "IMG/" + full_path.split('/')[-1]

def test() :
    #%%
    cfg = {"dropout_d1" : 0.6, 
           "dropout_cs" : 0.4 }
    
    model = build_model( cfg )
    model.summary() 
    #%%

if os.name == "posix" : 
    main() 
