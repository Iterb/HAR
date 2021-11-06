FROM tensorflow/tensorflow:latest-gpu

RUN apt update && apt install -y libcurl4-openssl-dev pkg-config libjsoncpp-dev zip \
    ffmpeg libsm6 libxext6 
RUN pip3 install keras numpy yacs seaborn wandb seaborn opencv-python scipy sklearn
RUN pip3 install -U tf2onnx
RUN pip3 install -U pip
RUN pip3 install onnx onnxruntime


