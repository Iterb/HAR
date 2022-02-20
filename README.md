# Code for a master's thesis at WUT
 
PL: Modelowanie i rozpoznawanie interakcji pary osób w sekwencji wideo z wykorzystaniem rekurencyjnych sieci neuronowych

ENG: Modeling and recognizing human interaction between two individuals
in a video sequence using recurrent neural networks

## Running 
Build both docker images: Dockerfile -> for inference and Dockerfile2 -> for model training. In the appropriate containers run:
```
python3 tools/train_net.py
```
for training models,
```
python3 tools/run_inference.py
```
for inference on wideo,
```
python3 tools/sweep_net.py
```
for optimizing hiperparametres,


All run arguments are taken from config file `defaults.py`.

## Dependencies 

[Link to csv data](https://drive.google.com/drive/folders/1EQwLbfHGTGeWVWAYojyO05M4DUaQdP_b?usp=sharing). 

[Link to trained onnx models](https://drive.google.com/file/d/1Ev2oIZn4I5DYS8nHSbrn43ERiT9zTBeg/view?usp=sharing). 
