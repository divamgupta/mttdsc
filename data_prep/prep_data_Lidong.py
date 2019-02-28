#!/usr/bin/env python
# coding: utf-8

# Prepering the data 



# utils for processing data of TD sentiment and other tasks

import json
import h5py
import numpy as np
import glob
import os


from twtokenize import tokenize
from keras.utils import to_categorical

from dlblocks import text
from dlblocks.pyutils import mapArrays , loadJson , saveJson , selectKeys , oneHotVec , padList
from dlblocks.pyutils import int64Arr , floatArr




def tok_and_replace( sen , en ):
	tw = sen.lower()
	target = en.lower()

	tw = tw.replace('$t$', target) #using already preprocessed data from Tang et al. 2016
	tw=tw.replace(target,' '+target+' ')
	tw=tw.replace(''.join(target.split()),' '+'_'.join(target.split())+' ')
	tw=tw.replace(target,' '+'_'.join(target.split())+' ')
	tweet = tokenize(tw)
	targetEn =  '_'.join(target.split()) 
	assert ( targetEn in tweet )
	return tweet , targetEn


# note that there is a bug in the below method
# d is being used for diffrent functions at diffrrent places
def makeTokenised( data ):
	ret = []
	for d in data:
		for dd in d['entities']:
			sentiment = dd['sentiment']
			tokensied , en  = tok_and_replace( d['sentence'] , dd['entity']  )
			L = { "tokens":[] , "sentence":d['sentence'] }
			entityPosition = -1
			for i , tok in enumerate(tokensied) :
				d =  {  "word":tok , "is_entitiy":False }
				if tok == en :
					d['is_entitiy'] = True
					entityPosition = i
					d['sentiment'] = sentiment 
				L['tokens'].append( d )
			L['sentiment'] = sentiment
			L['entityPosition'] = entityPosition
			L['entity'] = en
		ret.append( L )
	return ret 


def makeVocab( tokenisedData ):
	vocab = text.Vocabulary()
	for x in tokenisedData:
		for tok in x['tokens']:
			vocab.add_word( tok['word'] )
	return vocab





def getlidong( split ):
    if split=='train':
        lines = open("./lidong/training/target.train.raw.raw").read().split("\n")
    elif split == 'test':
        lines = open("./lidong/testing/target.test.raw.raw").read().split("\n")
    lines = map( lambda x : x.strip() , lines )
    train = []
    sentences = lines [ 0:: 3]
    entities = lines[1::3]
    sentiments = lines[2::3]
    sentiments = map( int , sentiments )
    dset = zip( sentences , entities , sentiments )
    assert len( sentiments)==len(entities) and len( entities)==len(sentences)
    dset = map( lambda x : {"sentence":x[0] , "entities":[{"entity": x[1] , "sentiment" : x[2] }]} , dset )
    return dset





train = makeTokenised( getlidong( 'train') )
test = makeTokenised( getlidong( 'test') )



vocab = makeVocab( train )



maxSentenceL = max( map( len , selectKeys( train , 'tokens') ))



print "maxSentenceL " , maxSentenceL 
print "vocab length" , len( vocab)


# # making the H5 file




import h5py

gf = h5py.File("../data/glovePrepped.h5")

import json
glovevocab = json.loads( gf['twitter_100_vocab'].value )
gloveVecs = np.array( gf['twitter_100_vecs'] )
def gloveDict( w ):
    if w in glovevocab:
        return glovevocab[w]
    else:
        return 0




print "glove vocal len" , len( glovevocab )
print gloveVecs.shape



def vecc( d ):
    ret = {}
    words , isEntitys = selectKeys( d['tokens'] , ('word' , 'is_entitiy') )
    enPos = words.index( d['entity'])
    assert isEntitys[enPos] == True
    entityWords = d['entity'].split("_")

    leftWords = words[ : d['entityPosition']  ]
    rightWords = words[ d['entityPosition'] +1 : ]
    words = leftWords + entityWords + rightWords
    entityMask = [0]*len( leftWords) + [1]*len(entityWords) + [0]*(len(rightWords))

    wordids = map( vocab , words )
    wordIdsLeft = map( vocab , leftWords )
    wordIdsRight = map( vocab , rightWords )

    ret['sentence'] = int64Arr( padList( wordids , maxSentenceL , 0 , 'left') )
    ret['entity_mask'] = int64Arr( padList( entityMask , maxSentenceL , 0 , 'left') )
    ret['sentence_left']  = int64Arr( padList( wordIdsLeft , maxSentenceL , 0 , 'left') )
    ret['sentence_right']   = int64Arr( padList( wordIdsRight[::-1] , maxSentenceL , 0 , 'left') )

    wordids = map( gloveDict , words )
    wordIdsLeft = map( gloveDict , leftWords )
    wordIdsRight = map( gloveDict , rightWords )
    wordIdsEntities = map( gloveDict , entityWords )

    ret['sentence_glove'] = int64Arr( padList( wordids , maxSentenceL , 0 , 'left') )
    ret['sentence_left_glove']  = int64Arr( padList( wordIdsLeft , maxSentenceL , 0 , 'left') )
    ret['sentence_right_glove']   = int64Arr( padList( wordIdsRight[::-1] , maxSentenceL , 0 , 'left') )

    ret['sentence_glove_vecs'] = gloveVecs[ int64Arr( padList( wordids , maxSentenceL , 0 , 'left') )]
    ret['sentence_left_glove_vecs']  = gloveVecs[int64Arr( padList( wordIdsLeft , maxSentenceL , 0 , 'left') )]
    ret['sentence_right_glove_vecs']   = gloveVecs[int64Arr( padList( wordIdsRight[::-1] , maxSentenceL , 0 , 'left') )]
    
    # In this we also include the entite in both left and right
    ret['sentence_left2_glove_vecs']  = gloveVecs[int64Arr( padList( wordIdsLeft+wordIdsEntities , maxSentenceL , 0 , 'left') )]
    ret['sentence_right2_glove_vecs']   = gloveVecs[int64Arr( padList( wordIdsRight[::-1]+wordIdsEntities[::-1] , maxSentenceL , 0 , 'left') )]
    ret['sentence_left2_glove']  = int64Arr( padList( wordIdsLeft+wordIdsEntities , maxSentenceL , 0 , 'left') )
    ret['sentence_right2_glove']   = int64Arr( padList( wordIdsRight[::-1]+wordIdsEntities[::-1] , maxSentenceL , 0 , 'left') )
    
    ret['sentence_right2_len'] = len( wordIdsRight )+len(wordIdsEntities)
    ret['sentence_left2_len'] = len( wordIdsLeft )+len(wordIdsEntities)
        
    ret['sentence_entity_glove']   = int64Arr( padList( wordIdsEntities , 20  , 0 , 'left') )
    ret['sentence_entity_glove_rightpad']   = int64Arr( padList( wordIdsEntities , 20  , 0 , 'right') )
    ret['sentence_entity_glove_vecs']   = gloveVecs[int64Arr( padList( wordIdsEntities , 5  , 0 , 'left') )]
    ret['sentence_entity_len']   = len( wordIdsEntities )

    entityWordIds =  map( gloveDict , entityWords ) 
    entityVec = gloveVecs[ entityWordIds ].mean( axis=0  )
    ret['entity_vec_glove'] = entityVec


    ret['sentiment_val'] =  floatArr( d['sentiment'] )
    ret['sentiment_id'] =  int64Arr( d['sentiment'] + 1 )
    ret['sentiment_onehot'] =  floatArr( oneHotVec( d['sentiment']+1 , 3  ) )

    return ret




train_arr = mapArrays( train , vecc )
test_arr = mapArrays( test , vecc )




import h5py



f = h5py.File("../data/tdlstm_prepped_1.h5" , "w")
f.create_group("train")
for k in train_arr.keys():
    f['train'].create_dataset( k , data=train_arr[k ])
f.create_group("test")
for k in test_arr.keys():
    f['test'].create_dataset( k , data=test_arr[k ])
f.close()




print "Done"

