FROM python:3.8.2-slim-buster

RUN pip install --upgrade pip
RUN pip install numpy scipy matplotlib pandas scikit-image scikit-learn ipython jupyter seaborn mlflow
RUN pip install jax jaxlib
RUN pip install lime tensorboard torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html