from keras.models import load_model
import json
import numpy as np


# download 16en_UK2_no_ent.h5 and 16en_lidong_mttdsc_deploy_vocab.json from :
# https://drive.google.com/file/d/1XlFd1Nfl_83VEk5aIvXX0h54AG5hTpAU/view?usp=sharing
# https://drive.google.com/open?id=1iEfZj3NEEC6Ma6s9voWSvGEdqoAl8mPm

# you need keras 2.1.4 , tensorflow 1.4.0 and python2 to run this

m = load_model('16en_UK2_no_ent.h5')
glovevocab = json.loads( open().read('16en_lidong_mttdsc_deploy_vocab.json') )


def get_sentiment( sent_l , target , sent_r  ):

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
        
    return m.predict( X )



if __name__ == '__main__':

    # example for sentence with target  taylor swift : i like taylor swift and her music is great

    sent_l = 'i like'
    target = 'taylor swift'
    sent_r = 'and her music is great'

    print get_sentiment( sent_l , target , sent_r  )
