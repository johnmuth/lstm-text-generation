# lstm-text-generation

This repo is a learning exercise for me. Machine learning learning.

The basic idea is: 
  - train a model on an example text, then
  - use the model to generate new text

I am a complete machine learning dilettante; everything in this repo is shamelessly stolen from http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

## getting started: environment 

> Large deep learning models require a lot of compute time to run. You can run them on your CPU but it can take hours or days to get a result. If you have access to a GPU on your desktop, you can drastically speed up the training time of your deep learning models.

If it's still available, I recommend using the AMI created by the author of the blog posts I'm stealing all this from, by following the instuctions here: http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/

## getting started: prerequisites

- Python 2

```
git clone git@github.com:johnmuth/lstm-text-generation.git
cd lstm-text-generation
pip install -r requirements.txt
```

