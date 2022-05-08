# 6.835 Final Project: Emotion Detection in Telehealth (EDiTH)
To run EDiTH, you will need to run this on an Apple M1 chip equipped computer, which comes with GPU. This system was implemented on a 2020 MacBook Pro M1 with 8GB memory.


## Prerequisite Setup Steps
0. Clone this repository onto your computer. On a Mac terminal, navigate to the directory you want and enter the following:
```bash
git clone https://github.com/jinwkim/emotion-detection.git
```

1. Install "conda env" by following instructions for "arm64: Apple Silicon" at the following link:
https://developer.apple.com/metal/tensorflow-plugin/

2. After installing conda, create a virtual conda environment running Python 3.9. The following script will create one named "6835" and activate the virtual environment:
```bash
conda create --name 6835 python=3.9
conda activate 6835
```

3. Install hdf5 from brew, which is necessary to install TensorFlow on M1 Mac's
```bash
brew install hdf5
```

4. Install TensorFlow following specific steps for M1 Mac's listed from the following link:
https://developer.apple.com/metal/tensorflow-plugin/
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

5. Install the necessary Python packages using pip and conda, in the following order:
```bash
pip install numpy
pip install opencv-python
pip install scipy
conda install cmake
pip install dlib
brew install portaudio --HEAD

pip install pyaudio --global-option="build_ext" --global-option="-I/opt/homebrew/include" --global-option="-L/opt/homebrew/lib"

conda install llvmlite
conda install -c conda-forge librosa
pip install pydub
pip install noisereduce
```

## To run EDiTH
In order to EDiTH, you need to have 2 terminals open, both activated with the conda virtual environment named "6835".

On the first terminal, run -- this would be the patient's side:
```python
python transmitter.py
```

On the second terminal, run -- this would be the physician's side:
```python
python receiver.py
```

You will see 2 windows pop up both showing the video stream captured by your computer's webcam. One window is the unedited raw video stream, and the other window titled "RECEIVER" is the resulting raw video that would have been streamed over to the other computer (but in EDiTH, it was streamed to localhost, back to your computer).

When EDiTH detects negative emotions from facial expressions or speech over a ~11-second span, it should show a visual notification and an audible alert sound. Make sure your "Do Not Disturb" mode is disabled.

## To create and train your own model
You can download fer2013.csv from https://gitlab.com/andwong/mp4_data/-/tree/master/kaggle_fer2013 and put into the folder called "face-emotion"

In terminal, make sure you are in the "face-emotion" folder. You can run the following line of code that should generate a ".h5" model in the "models" directory:
```python
python face-emotion-model.py
```

You can also modify the model being created by altering the "create_model()" method.