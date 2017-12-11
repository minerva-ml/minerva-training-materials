# Welcome to Minerva
CHECK THE VIDEO

[![Minerva welcome video](https://i.ytimg.com/vi/bI-YlVJJl-I/2.jpg)](https://www.youtube.com/watch?v=bI-YlVJJl-I&feature=youtu.be)

Minerva is a platform that lets you learn real life data science at your own pace, piece by piece, feature at a time.
We re-implement solutions to the most difficult data scientific problems so that you can learn how to do that too.
Every problem is a collection of small, digestable tasks. Every task is a piece of code, a step in a large machine learning pipeline. You just need to create your implementation!

We know what the final score for a well implemented pipeline should be. So as you solve tasks and re-implement parts of the pipeline we will be checking whether your implementation does the job well enough to keep the score high. So beware, we are watching :)

# What is it

With Minerva you learn by reproducing real life complicated machine learning pipelines like this:

![fashion_mnist](.png)


or this:

![whales](.png)

<img src="img/doom.gif" alt="Doom Health Gathering" width="265" height="200"/><img src="img/minitaur.gif" alt="PyBullet Minitaur" width="265" height="200"/> <img src="img/ant.gif" alt="Gym Extensions Ant" width="250" height="200"/>


You achieve that by replacing this pipeline with your code. 
For example in one of the tasks you are asked to build a model:

```python
    CONFIG={'input_size':28, 'classes':10}
    
    def solution(input_size, classes, **kwargs):
        input_image = Input(shape=(input_size, input_size, 1))
        """
        Build your keras model here
        """
        class_prediction = Dense(classes, activation='softmax', name='output')(x)
        model = Model(input_image, class_prediction)
        return model

``` 
You write your code and execute a simple bash command to submit your solution:
```bash
python run minerva.py submit --task_nr 1 --filepath user_solution/notebooks/task1.ipynb
```
or if you want to use neptune you go:
```bash
neptune run run minerva.py -- submit --task_nr 1 --filepath user_solution/notebooks/task1.ipynb
```

The pipeline fits all the steps that were modified by your code, it evaluates the score and voila you get your feedback.

# Installation
This project assumes `python 3`.
The rest of the requirements can be installed via:
```bash
pip install -r requirements.txt
```

In order to use neptune monitoring please login via
```bash
neptune login
```
Once you are logged in create a project called Minerva. If you want to choose a different name go remember to change the project key in the neptune_config.yaml.
