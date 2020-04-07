import numpy as np

from network import create_model

from threading import Thread

from json import loads, dumps

from kafka import KafkaConsumer, KafkaProducer

from flask import Flask


app = Flask(__name__)

filename = 'weights-ep2076-val_loss0.2518.hdf5'

net = create_model(training=False)

net.load_weights(f'/tmp/src/Ascalon.NeuralNetwork.Service/ckpt/{filename}')

producer = KafkaProducer(bootstrap_servers=['kafka:9092'],
                         value_serializer=lambda x:
                         dumps(x).encode('utf-8'))

consumer = KafkaConsumer(
            'neuralnetwork_data',
            bootstrap_servers=['kafka:9092'],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            partition_assignment_strategy=RoundRobinPartitionAssignor,
            group_id='NeuralNetworkService',
            value_deserializer=lambda x: loads(x.decode('utf-8')))

def predict(x):
    pred = net.predict(x)
    label = np.argmax(pred, axis=1)  # [0]
    return label


def KafkaConsumer():
    while True:
        for message in consumer:
            tasks = []
            label = 0
            id = 0
            for data in message.value:
                task = [float(data['Gfx']), float(data['Gfy']), float(data['Gfz']),
                        float(data['Wx']), float(data['Wy']), float(data['Speed']),
                        float(data['Wz'])]
                label = float(data['Label'])
                id = data['Id']
                tasks.append(task)
            result = np.reshape(tasks, (-1, 50, 7))
            for element in result:
                element[:] = element - element.mean(axis=0)
            something = predict(result)
            data = {'result': int(something[0]+1), 'label': label, 'id': id}
            producer.send('client_service_data', value=data)


kafkaConsumer = Thread(target=KafkaConsumer)
kafkaConsumer.start()

if __name__ == '__main__':
    app.run(debug=True)
