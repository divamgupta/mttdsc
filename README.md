# Multi-task Learning for Target-dependent Sentiment Classification

Packaged datasets and Keras code for the paper [Multi-task Learning for Target-dependent Sentiment Classification](https://arxiv.org/abs/1902.02930).

We use `tensorflow-gpu-1.4.0` which needs `cudnn6`.  To run on CUDA ca 2019, you need to [download cudnn6 from here](http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6_6.0.21-1+cuda8.0_amd64.deb) and install along with CUDA8.

Prepare a virtual environment and install requirements as follows.
```shell
$ virtualenv -p `which python2` /path/to/mttdsc
$ source /path/to/mttdsc/bin/activate
(mttdsc)$ pip install -r requirements.txt
```

We will assume this code has been cloned to `/path/to/mttdsc` as the code base directory.  Download the [zipped data files](https://drive.google.com/open?id=18av-HZCx1G14CURRyYrjjmNcevOphaQc) and unzip in the code base directory, which will place all the .h5 files in the data subdirectory.  [Gdrive](https://github.com/prasmussen/gdrive) can be used for downloading.
```bash
$ cd /tmp
$ gdrive download 18av-HZCx1G14CURRyYrjjmNcevOphaQc
$ cd /path/to/mttdsc
$ unzip /path/to/zipfile
```

## Using the pretrained model

We have provided an easy to use API to get the the sentiment of any sentence. The API automatically downloads the model weights. Refer to the following snippet to use our pretrained sentiment model. 

```python
from tdsc_pretrained import get_sentiment
print get_sentiment( 'i like taylor swift and her music is great'  , 'taylor swift'  )
```

