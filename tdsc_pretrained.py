from keras.models import load_model
import json
import numpy as np
import keras

# download 16en_UK2_no_ent.h5 and 16en_lidong_mttdsc_deploy_vocab.json from :
# https://drive.google.com/file/d/1XlFd1Nfl_83VEk5aIvXX0h54AG5hTpAU/view?usp=sharing
# https://drive.google.com/open?id=1iEfZj3NEEC6Ma6s9voWSvGEdqoAl8mPm

# you need keras 2.1.4 , tensorflow 1.4.0 and python2 to run this



model_url = 'https://github.com/divamgupta/mttdsc/releases/download/weights_uk2/16en_UK2_no_ent.1.h5'
vocab_url = 'https://github.com/divamgupta/mttdsc/releases/download/weights_uk2/16en_lidong_mttdsc_deploy_vocab.1.json'


m = load_model( keras.utils.get_file( '16en_UK2_no_ent.h5' , model_url  ) )
glovevocab = json.loads( open( keras.utils.get_file( '16en_lidong_mttdsc_deploy_vocab.json' , vocab_url  )  ).read() )



def get_sentiment( sent , target ):

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
