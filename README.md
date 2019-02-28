# Multi-task Learning for Target-dependent Sentiment Classification

Packaged datasets and Keras code for the paper [Multi-task Learning for Target-dependent Sentiment Classification](https://arxiv.org/abs/1902.02930).

We use `tensorflow-gpu-1.4.0` which needs `cudnn6`.  To run on CUDA ca 2019, you need to [download cudnn6 from here](http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6_6.0.21-1+cuda8.0_amd64.deb) and install along with CUDA8.

Prepare a virtual environment and install requirements as follows.
```shell
$ virtualenv -p `which python2` /path/to/mttdsc_env
$ source /path/to/mttdsc_env/bin/activate
(mttdsc_env)$ pip install -r requirements.txt
```

We will assume this code has been cloned to `/path/to/mttdsc` as the code base directory.  Download the [zipped data files](https://drive.google.com/open?id=18av-HZCx1G14CURRyYrjjmNcevOphaQc) and unzip in the code base directory, which will place all the .h5 files in the data subdirectory.  [Gdrive](https://github.com/prasmussen/gdrive) can be used for downloading.
```bash
$ cd /tmp
$ gdrive download 18av-HZCx1G14CURRyYrjjmNcevOphaQc
$ cd /path/to/mttdsc
$ unzip /path/to/zipfile
```

If you want to prepare the data sets by yourself,

```shell
(mttdsc_env)$ cd /path/to/mttdsc/data_prep
(mttdsc_env)$ cd python prep_data_Lidong.py
```
## Training the models

To train the models use train.py . Refer to the following commands 

```shell
(mttdsc_env)$ python train.py config_files/tdgru_lidong.json   # training TDGRU on Lidong Dataset
(mttdsc_env)$ python train.py config_files/naive_mtl_lidong.json   # training NaiveMTL on Lidong Dataset
(mttdsc_env)$ python train.py config_files/mttdsc_lidong.json   # training MTTDSC on Lidong Dataset
```

Test results would be printed towared the end of completion of the script. 

UK Election dataset will be added shortly. 

## Using the pretrained model

We have provided an easy to use API to get the the sentiment of any sentence. The API automatically downloads the model weights. Refer to the following snippet to use our pretrained sentiment model. 

```python
from tdsc_pretrained import get_sentiment
print get_sentiment( 'i like taylor swift and her music is great'  , 'taylor swift'  )
```

