
from models import *
import sys
import json


if len( sys.argv ) < 2:
    print "Usage: python train.py path/to/config.json"

    
config_path = str(sys.argv[1])
config = json.loads(open( config_path ).read())

print "training " ,  config['model_name']

m_class = locals()[ config['model_name'] ] 

model = m_class( exp_location="outputs" , config_args = config )
model.train()





