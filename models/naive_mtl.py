from keras.models import *
from keras.layers import *
import keras
from dlblocks.keras_utils import allow_growth , showKerasModel
allow_growth()
from dlblocks.pyutils import env_arg
import tensorflow as tf

from Utils import Trainer





class NaiveMTL(Trainer):


    def build_model(self):

        gloveSize = 100
        vocabSize = 1193515

        #nEn = env_arg("nEn" , 16 , int  )

        maxSentenceL = self.config['maxSentenceL'] # self.config['maxSentenceL']
        rnn_type = self.config['rnn_type'] 
        nHidden = self.config['nHidden'] #env_arg("nH" , 64 , int  )
        opt_name = 'adam'
        #batch_size = self.config['batch_size'] # env_arg("batch_size" , 64 , int  )
        dropout = self.config['dropout'] # env_arg("drop" , 0.2 , float  )
        recurrent_dropout = self.config['recurrent_dropout'] #  env_arg("rec_drop" , 0.2 , float  )
        lr = -1 # env_arg("lr" , -1.0 , float  ) # default keras
        # epochs =  3# env_arg("epochs" , 3 , int  )
        #exp_name =  env_arg("exp_name" , "naive_mtl" , str  )


        # Loading the weights of the pre trainid Model

        if rnn_type == 'lstm':
            rnn = LSTM
        elif rnn_type == 'gru':
            rnn = GRU

        inp = Input(( maxSentenceL ,  ) ) # left
        inp_x = inp

        embed = (Embedding( vocabSize , gloveSize ,   trainable=False )  )


        inp_x = embed ( inp_x )


        inp_rev = Lambda(  lambda x:K.reverse(x,axes=1)  )( inp_x) # right

        rnn_left = rnn( nHidden , return_sequences=True , dropout=dropout , recurrent_dropout=recurrent_dropout )
        rnn_right = rnn( nHidden , return_sequences=True , dropout=dropout , recurrent_dropout=recurrent_dropout )


        left_x = rnn_left( inp_x )
        right_x = rnn_right( inp_rev  )
        right_x  = Lambda(  lambda x:K.reverse(x,axes=1)  )( right_x )

        c_x = Concatenate( axis=-1 )([ left_x ,right_x] )

        c_x = GlobalAvgPool1D()( c_x )
        x = Dense( 3 )( c_x )
        out = Activation('softmax')( x )

        m = Model( inp  , out )
        m.load_weights( "./data/lr_lstm_glove_3.2ft_2_ep0.h5" )



        def getM():

            # setting the weights from the pretrained model to the new model
            if rnn_type == 'lstm':
                rnn = LSTM
            elif rnn_type == 'gru':
                rnn = GRU


            rnn_left_ = rnn( nHidden , return_sequences=True , dropout=dropout , recurrent_dropout=recurrent_dropout , trainable=False  )
            rnn_right_ = rnn( nHidden , return_sequences=True , dropout=dropout , recurrent_dropout=recurrent_dropout  , trainable=False )

            rnn_left_2 = rnn( nHidden , return_sequences=True , dropout=dropout , recurrent_dropout=recurrent_dropout , trainable=True  )
            rnn_right_2 = rnn( nHidden , return_sequences=True , dropout=dropout , recurrent_dropout=recurrent_dropout  , trainable=True )



            def getPrimModel(): # will return a submodel with the shard weights and both the LSTMs

                left_i = Input(( maxSentenceL ,  ) )
                right_i = Input(( maxSentenceL ,  ) )

                left_x = left_i
                right_x = right_i

                left_x = embed ( left_x )
                right_x = embed ( right_x )

                left_x_1 = rnn_left_( left_x)
                right_x_1 = rnn_right_( right_x )

                rnn_left_.set_weights( rnn_left.get_weights() )
                rnn_right_.set_weights( rnn_right.get_weights() )

                left_x_1 = Lambda(lambda x:x[:,-1,:])( left_x_1 ) # coz return seq true
                right_x_1 = Lambda(lambda x:x[:,-1,:])( right_x_1 )


                x =  Concatenate()([ left_x_1 , right_x_1 ])    
                x = Dense( 3 )( x )
                out = Activation('softmax')( x )

                return left_i , right_i , out


            def getAuxModel() : 

                inp = Input(( maxSentenceL ,  ) ) # left
                inp_x = inp
                inp_x = embed ( inp_x )
                inp_rev = Lambda(  lambda x:K.reverse(x,axes=1)  )( inp_x) # right

                left_x = rnn_left_( inp_x )
                right_x = rnn_right_( inp_rev  )
                right_x  = Lambda(  lambda x:K.reverse(x,axes=1)  )( right_x )

                rnn_left_.set_weights( rnn_left.get_weights() )
                rnn_right_.set_weights( rnn_right.get_weights() )

                c_x = Concatenate( axis=-1 )([ left_x ,right_x] )
                c_x = GlobalAvgPool1D()( c_x )
                x = Dense( 3 )( c_x )
                out = Activation('softmax')( x )

                return inp ,out 



            left_i_prim , right_i_prim , out_prim = getPrimModel()
            inp_aux , out_aux = getAuxModel()


            m = Model([  left_i_prim , right_i_prim  ,  inp_aux ] , [ out_prim ,out_aux  ] )


            if lr > 0:
                opt = getattr( keras.optimizers , opt_name )(lr= lr ) 
            else:
                opt = getattr( keras.optimizers , opt_name )() 

            m.compile( opt , 'categorical_crossentropy' , metrics=['accuracy'])

            return m

        self.model = getM()
        Trainer.build_model( self  )
        

    
# config = {}
# config['maxSentenceL'] = 35
# config['nHidden'] = 64
# config['dropout'] = 0.2
# config['recurrent_dropout'] = 0.2
# config['epochs'] = 6
# config['batch_size'] = 64
# config['rnn_type'] = 'gru'
# config['nEn'] = 16
# config['dataset'] = "./data/tdlstm_prepped_1.h5"
# config['n_samp'] = 18744
# config['n_tasks'] = 2

# model = NaiveMTL( exp_location="/tmp" , config_args = config )
# model.train()

