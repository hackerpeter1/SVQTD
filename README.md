For doing research, **Data Request** instructions are in our page [here](hackerpeter1.github.io/SVQTD/).

### Dataset preparation
* 1. download youtube videos with a python script and convert to audios using [ffmpeg](https://ffmpeg.org/)
* 2. performing music source separation based on [spleeter](https://github.com/deezer/spleeter)
* 3. energy-based segmentation, reference code can be found in ./split.py
* 4. extracting feature set using [OPENSMILE](https://audeering.github.io/opensmile/) (optional, only if you are interested in training with traditional feature set)

### Training files
* 1. Some pooling method for recognition neural network can be found in ./modules.
* 2. Some models are in ./models.
* 3. Some config files for respectively training Transformer and ResNet are in ./config. 
* 4. ./E2E.py can be used to train neural networks based on config files.
* 5. ./RPSVM.py can be used to extract embeddings and train a SVM classifier using them. 
* 6. ./FSSVM.py can be used to train a SVM classifier using features from ComParE feature set.

Since our code is not user-friendly, if you have any questions about dataset downloading or the training code, please feel free to contact me through yanze.xu@outlook.com. Also welcome to talk with me if you are interested in timbre phenoemena. 

