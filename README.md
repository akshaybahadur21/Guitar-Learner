# Guitar-Learner &nbsp; 🎸
[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Autopilot/blob/master/LICENSE.txt)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)

A guitar chord detection and classifier for humans

## Code Requirements 🦄
You can install Conda for python which resolves all the dependencies for machine learning.

##### pip install requirements.txt

## Description 🎼
A guitar chord is a set of notes played on a guitar. A chord's notes are often played simultaneously, but they can be played sequentially in an arpeggio. The implementation of guitar chords depends on the guitar tuning. 

Most guitars used in popular music have six strings with the "standard" tuning of the Spanish classical guitar, namely E-A-D-G-B-E' (from the lowest pitched string to the highest)

## Python  Implementation 👨‍🔬

**Supported Chords**

-  A
-  B
-  C
-  D
-  E
-  F
-  G

If you face any problem, kindly raise an issue

## Setup 🖥️

1) First, you have to create a hand-chord database. For that, run `CreateDataset.py`. Enter the gesture name and you will get 2 frames displayed. Look at the contour frame and adjust your hand to make sure that you capture the features of your hand. Press 'c' for capturing the images. It will take 1200 images of one gesture. Try moving your hand a little within the frame to make sure that your model doesn't overfit at the time of training.
2) Repeat this for all the features you want.
3) For training the model, run `Trainer.py`
4) Finally, run `GuitarLearner.py` for testing your model via webcam.
5) Play an awesome riff.🤩

## Execution 🐉

```
python3 GuitarLearner.py
```

## Results 📊

<img src="https://github.com/akshaybahadur21/BLOB/blob/master/final.gif">

## Future Scope 🔮
- Add Barre chords
- Localize and detect fret chords

## References 🔱
 
 -  Ivan Grishchenko and Valentin Bazarevsky, Research Engineers, Google Research. [Mediapipe by Google](https://github.com/google/mediapipe)
