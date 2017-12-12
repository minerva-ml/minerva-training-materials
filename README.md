# Minerva
Minerva is an educational project that lets you learn advanced data science on real-life, curated problems.

---

## Hands-on approach to learning
With Minerva you will reproduce, piece by piece, solution to the most difficult data scientific problems, especially challenges. Since each **problem** is quite complex, we have divided it into a collection of small self-contained pieces called **tasks**.

**Task** is a single step in machine learning pipeline, has its own learning objectives, descriptions and a piece of code that needs to be implemented. This is your job, create technical implementation that fulfill this gap. You use your engineering skills, extensive experimentation and our feedback, in order to make sure that your implementation meets certain quality level. We know what the final score for a well implemented pipeline should be. So as you solve tasks and re-implement parts of the pipeline we will be checking whether your implementation does the job well enough to keep the score high.

## Reproduce Kaggle winning solutions in a transparent way &rarr; learn advanced data science
Working on **tasks** that, if taken together, creates solution to the **problem** lets you reproduce Kaggle winning solution, piece by piece. This is our hands on approach to learning, because you can work on each part of the winning implementation by yourself.

## Available problems

| Problem        | Description   |
| -------------- | ------------- |
| Fashion mnist  | Get started with Minerva by solving easy pipeline on nice dataset [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist 'Fashion-MNIST dataset') |
| Whales         | Reproduce [Right Whale Recognition](https://www.kaggle.com/c/noaa-right-whale-recognition 'Right Whale Recognition') Kaggle winning solution! |
| | *(more problems will be published in the future, so stay tuned)* |

---

# Installation
This project assumes python 3.5
1. Clone or download minerva code: `git clone https://github.com/neptune-ml/minerva.git`

### CUDA
Note, that it is highly recommended to train your solution on GPU.
1. Check your CUDA version. You can get this information by running `nvcc --version` or `cat /usr/local/cuda/version.txt` or by asking your system administrator.
2. install TensorFlow `pip3 install tensorflow-gpu==1.2.0`
3. install appropriate PyTorch, by following [PyTorch Get Started](http://pytorch.org/)
4. install remaining requirements `pip3 install -r requirements.txt`

### no CUDA
In case you do not have GPU capability, you will work on your processor. Also, you may want to consider using [Neptune](https://neptune.ml 'Machine Learning Lab') to train your models.

1. install TensorFlow `pip3 install tensorflow==1.2.0`
2. install PyTorch, by following [PyTorch Get Started](http://pytorch.org/)
3. install remaining requirements `pip3 install -r requirements.txt`

---

## User support
You can seek support in two ways:
1. check [Minerva wiki](https://github.com/neptune-ml/minerva/wiki 'Minerva wiki')
2. consider adding issue with label `question`, `bug` or `feature request`. Note, that issues are for technical problems.

## Contributing to Minerva
You are very welcome to contribute your piece to Minerva. There are three main ways to do so:
1. Use GitHub issues
2. Create Pull Request
3. With custom ideas, contact us directly on [minerva@neptune.ml](minerva@neptune.ml 'coming soon...')

## About the name
Minerva is a Roman goddess of wisdom, arts and craft. She was usually presented with the strong association with knowledge. Her sacred creature *'owl of Minerva'* symbolizes wisdom and knowledge. We think that this name depicts our project very well, since it is about acquiring knowledge and skills.
