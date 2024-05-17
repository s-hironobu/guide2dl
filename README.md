## The Engineer's Guide to Deep Learning

This is the repository of [The Engineer's Guide to Deep Learning](https://www.interdb.jp/dl/).

### Developing Environment

```
Python				3.9.13
conda				22.11.1
matplotlib			3.6.0
numpy				1.23.3
scikit-learn 			1.2.0
keras				2.10.0
tensorflow-macos		2.10.0
tensorflow-metadata		1.13.1
tensorflow-metal		0.6.0
```  

To ensure compatibility, please create the environment using the above versions before running the program.

### Note

- I am happy to receive bug reports and assist with bug fixes.
- I am unable to provide support for any installation issues.

### Part 0: [Basic Knowledge](./Part-00/)


### Part 1: [Basic Neural Networks](./Part-01/)


### Part 2: [Recurrent Neural Networks](./Part-02/)


### Part 3: [Natural Language Processing and Attention Mechanisms](./Part-03/)


### Part 4: [Transformer](./Part-04/)


### License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


### Appendix: Installing TensorFlow on M1 Mac Using Miniconda

Installing TensorFlow and its related modules is highly version-dependent, which can make it challenging to ensure everything works together smoothly.


Here's how I installed it on M1 Mac. While this installs a slightly older version, it was up-to-date at the time of my development.


#### [1] Installing miniconda

```
$ mkdir -p ~/miniconda3
$ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm -rf ~/miniconda3/miniconda.sh
$ ~/miniconda3/bin/conda init bash
```

After that, close this terminal, and then create another terminal.

#### [2] Installing TensorFlow

```
$ conda activate
$ cd miniconda3/
$ conda install python==3.9.13
$ pip install numpy==1.23.3
$ conda install -c apple tensorflow-deps==2.7.0
$ pip install tensorflow-macos==2.10.0
$ pip install tensorflow-metal==0.6.0
```

```
$ python
>>> import tensorflow
>>> # No error messages indicate successful installation
```

#### [3] Installing Other modules

```
$ pip install keras==2.10.0
$ pip install matplotlib==3.6.0
$ pip install sklearn==1.4.2
```

