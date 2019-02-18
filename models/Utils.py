
from keras.models import *
from keras.layers import *

import keras
import keras.backend as K
from keras import initializers

import h5py

from sklearn.metrics import classification_report , accuracy_score , f1_score , recall_score , precision_score , mean_absolute_error , mean_squared_error


from dlblocks.keras_utils import WeightsPredictionsSaver
from dlblocks.TrainUtils import BaseTrainer
import random
import json
import os
import numpy as np



import numpy as np
import pickle


# data -> 18av-HZCx1G14CURRyYrjjmNcevOphaQc


def free_tf_mem():
    import keras.backend.tensorflow_backend
    if keras.backend.tensorflow_backend._SESSION:
        import tensorflow as tf
        tf.reset_default_graph()
        keras.backend.tensorflow_backend._SESSION.close()
        keras.backend.tensorflow_backend._SESSION = None


def evel( GT , PR ):
    d = {}
    # print classification_report( GT , PR , digits=5)
    d[ "accuracy" ] =   accuracy_score( GT , PR )
    d[ "f1" ] = f1_score( GT , PR , average='macro' )
    d[ "f2 2class" ] =  (f1_score( GT , PR , average=None )[0] + f1_score( GT , PR , average=None )[2]) /2
    
    d[  "recall" ] = recall_score( GT , PR , average='macro' )
    d[ "recall 2 class " ] = (recall_score( GT , PR , average=None )[0] + recall_score( GT , PR , average=None )[2]) /2
    
    d[ "precision " ] =  precision_score( GT , PR , average='macro' )
    d[ "precision 2 class" ] =  (precision_score( GT , PR , average=None )[0] + precision_score( GT , PR , average=None )[2]) /2
    
    d[ "MAE " ] =  mean_absolute_error( GT , PR  )
    d[ "MSE " ] =   mean_squared_error( GT , PR   )
    
    return d






class Trainer( BaseTrainer ):
    """docstring for Trainer"""

    def __init__(self, **kargs ):
        BaseTrainer.__init__( self,  **kargs  )


    def set_dataset( self ):

        dataset_path = self.config['dataset']
        f = h5py.File( dataset_path , "r")
        f2 = h5py.File( "./data/sanders_sentiment.h5" , "r")
        
        maxSentenceL = self.config['maxSentenceL']
        n_test = int(f['test']['sentiment_onehot'].shape[0])
        
        tr_x = [ 
            np.concatenate([f['train']['sentence_left2_glove']]*5)[: self.config['n_samp']  , -maxSentenceL:   ],
            np.concatenate([f['train']['sentence_right2_glove']]*5)[: self.config['n_samp']  , -maxSentenceL:   ] ,
            np.concatenate([f2['train']['sentence_glove']]*5)[ : self.config['n_samp'] , -maxSentenceL: ]
        ]

        tr_y = [
            np.concatenate([f['train']['sentiment_onehot']]*5)[: self.config['n_samp'] ] , 
            np.concatenate([f2['train']['sentiment_onehot']]*5)[ : self.config['n_samp'] ]
        ][: self.config['n_tasks'] ]

        te_x = [ 
            np.concatenate([f['test']['sentence_left2_glove']]*5)[:n_test , -maxSentenceL:   ],
            np.concatenate([f['test']['sentence_right2_glove']]*5)[:n_test  , -maxSentenceL:   ] ,
            np.concatenate([f2['test']['sentence_glove']]*5)[ :  , -maxSentenceL: ]
        ]

        te_y = [
            np.concatenate([f['test']['sentiment_onehot']][ : n_test ] ) , 
            np.concatenate([f2['test']['sentiment_onehot']]*5)
        ][: self.config['n_tasks'] ]
        
        
        self.data_inp = tr_x
        self.data_target = tr_y
        self.te_data_inp = te_x
        self.te_data_data_target = te_y

        BaseTrainer.set_dataset( self  )


    def build_model(self):
        self.model.compile('adam' , 'categorical_crossentropy' , metrics=['accuracy'])
        BaseTrainer.build_model( self  )
        
        
        
    def train( self ):
        
        
        if not self.dataset_set:
            self.set_dataset()
            
        PR = []
        for en in range( self.config['nEn'] ):
            self.build_model()
            self.model.fit( self.data_inp , self.data_target , epochs=self.n_epochs , shuffle="batch" , batch_size=self.config['batch_size'] )
            if self.config['n_tasks'] > 1:
                pr = self.model.predict( self.te_data_inp )[0]
            else:
                pr = self.model.predict( self.te_data_inp )
            PR.append( pr )
            free_tf_mem()
            
        self.PR = np.mean(PR , axis=0)
        print "self.PR" , self.PR.shape
        self.evaluate()



    def evaluate( self ):
        #print "self.te_data_inp[0].argmax( axis=-1) " , self.te_data_inp[0].argmax( axis=-1)
        #print "self.PR.argmax( axis=-1)  ) " , self.PR.argmax( axis=-1)
        d = evel( self.te_data_data_target[0].argmax( axis=-1)  , self.PR.argmax( axis=-1)  )
        d['exp_name'] = self.exp_name
        s = "expname:results " + self.exp_name+":("
        for k in d:
            s += str(k) +":"+ str(d[k])+"|"
        s += ")"
        print s
        return d



