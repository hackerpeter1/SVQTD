### Please switch to page branch for dataset downloading.

### Dataset preparation
* 1. download youtube videos using python script and convert to audios using ffmpeg
* 2. performing music source separation based on spleeter
* 3. energy-based segmentation, code can be found in xxx (coming soon). 

### Training files
* 1. Some pooling method for recognition neural network can be found in ./modules.
* 2. Some models are in ./models.
* 3. Some config files for respectively training Transformer and ResNet are in ./config. 
* 4. ./E2E.py can be used to train neural networks based on config files.
* 5. ./RPSVM.py can be used to extract embeddings and train a SVM classifier using them. 
* 6. ./FSSVM.py can be used to train a SVM classifier using features from ComParE feature set.

If you have any questions about dataset downloading or the code, feel free to contact me through yanze.xu@outlook.com. Also welcome to talk with me if you are interested in timbre phenoemena. 

