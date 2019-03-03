from keras.models import load_model
import json
import numpy as np
import keras


# you need keras 2.1.4 , tensorflow 1.4.0 and python2 to run this


model_url = 'https://github.com/divamgupta/mttdsc/releases/download/weights_lidong/16en_lidong_mttdsc_deploy.h5'
vocab_url = 'https://github.com/divamgupta/mttdsc/releases/download/weights_lidong/16en_lidong_mttdsc_deploy_vocab.json'



m = load_model( keras.utils.get_file( '16en_lidong_mttdsc_deploy.h5' , model_url  ) )
glovevocab = json.loads( open( keras.utils.get_file( '16en_lidong_mttdsc_deploy_vocab1.json' , vocab_url  )  ).read() )



def get_sentiment( sent , target ):

    sent = sent.lower()
    target = target.lower()

    assert (target in sent) , "Target not found in the input sentence "
    sent_l = sent.split(target)[0].strip()
    sent_r = sent.split(target)[1].strip()

    L = (sent_l.split(' ') + target.split(' '))
    R = (target.split(' ') + sent_r.split(' '))[::-1]


    X = [  
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
       ,np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) ]

    for i,ll in enumerate(L[::-1]):
        if ll in glovevocab:
            X[0][0][-i] = glovevocab[ll]

    for i,ll in enumerate(R[::-1]):
        if ll in glovevocab: 
            X[1][0][-i] = glovevocab[ll]
        
    rr =  m.predict( X )[0]
    return {"neg": rr[0] , "nuet":rr[1] , 'pos':rr[2]}



if __name__ == '__main__':

    # example for sentence with target  taylor swift : i like taylor swift and her music is great

    print get_sentiment( 'i like taylor swift and her music is great'  , 'taylor swift'  )
