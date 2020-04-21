FROM python:3.7
WORKDIR /tmp
COPY /src ./src
RUN pip install pandas
RUN pip install tensorflow
RUN pip install numpy
RUN pip install kafka-python
CMD python /tmp/src/Ascalon.NeuralNetwork.Service/model/inference.py