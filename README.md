## The Engineer's Guide to Deep Learning

This is the repository of [The Engineer's Guide to Deep Learning](https://www.interdb.jp/dl/).

### Developing Environment

```
Python                       3.11.5
keras                        2.15.0
pip                          24.0
numpy                        1.26.4
matplotlib                   3.9.0
tensorflow                   2.15.1
tensorflow-metal             1.1.0
scikit-learn                 1.5.0
```  

To ensure compatibility, please create the environment using the above versions before running the program.

### Note

- I am happy to receive bug reports and assist with bug fixes.
- I am unable to provide support for any installation issues.


### Part 1: [Basic Neural Networks](./Part-01/)


### Part 2: [Recurrent Neural Networks](./Part-02/)


### Part 3: [Natural Language Processing and Attention Mechanisms](./Part-03/)


### Part 4: [Transformer](./Part-04/)


### Appendix: [Basic Knowledge](./Appendix/)


### License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


### Appendix: Installing TensorFlow on M1 Mac

Reference:
+ [Get started with tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)

Here's how I installed it on M1 Mac:

#### [1] Installing Python 3.11.5 using pyenv

```
$ pyenv install 3.11.5
$ pyenv global 3.11.5
```

#### [2] Installing TensorFlow under venv

```
$ python3 -m venv ~/venv-metal
$ source ~/venv-metal/bin/activate
$ python -m pip install -U pip
$ pip install tensorflow==2.15.1
$ pip install tensorflow-metal==1.1.0
$ pip install matplotlib==3.9.0
```

