# Neural network

To run you need to have a few dependencies installed (I recommend installing everything in a virtualenv):

* dlib python (available in pip)
* opencv 3.2 (compile from source with python support recommended)
* numpy
* sklearn
* matplotlib
* IPython (for data collection)
* pandas
* pickle
* Python >= 2.7.10

Start the server (while in your virtualenv) using:

```
python src/smile_server.py
```

If matplotlib complains a bunch on OSX, you should read http://matplotlib.org/faq/osx_framework.html

You should also probably create the directory `data/` as soon as you pull the repo.

`data/smile_home` is where faces and identities received through the api point `/train` will get saved. So take good care of it.

The first time you run the program, the LFW (face database) will be downloaded and it could take a long time.

Preprocessing images are cached so future runs of the app should be quicker than your first time.
