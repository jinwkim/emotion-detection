# Emotion Detection in Telehealth (EDiTH)
6.835 project

## Emotion from face expressions
### conda env setup
https://developer.apple.com/metal/tensorflow-plugin/
conda install -c apple tensorflow-deps
conda create --name 6835 python=3.9
conda activate 6835   
conda install pip

brew install hdf5
pip install tensorflow-macos
pip install tensorflow-metal

pip install opencv-python
pip install matplotlib
pip install scipy

### To create and train the model
Download fer2013.csv from https://gitlab.com/andwong/mp4_data/-/tree/master/kaggle_fer2013 and put into a new folder called "data"
Create a new folder called "result"
Run "python face-emotion.py" to generate model.h5 inside result folder

model from https://github.com/omar178/Emotion-recognition
Accuracy 66%

### To run the code
python transmitter.py - "patient"
python receiver.py - "doctor"

### Dependencies
opencv-python==4.5.5
tensorflow==2.8.0
numpy==1.22.3

## Emotion from speech
brew install portaudio --HEAD
pip install pyaudio --global-option="build_ext" --global-option="-I/opt/homebrew/include" --global-option="-L/opt/homebrew/lib"
conda install llvmlite
conda install -c conda-forge librosa

pip install pydub
pip install noisereduce