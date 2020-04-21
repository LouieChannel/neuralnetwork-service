import numpy as np

from network import create_model

from threading import Thread

from json import loads, dumps

from kafka import KafkaConsumer, KafkaProducer

from kafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor

filename = 'weights-ep2076-val_loss0.2518.hdf5'

net = create_model(training=False)

net.load_weights(f'/tmp/src/Ascalon.NeuralNetwork.Service/ckpt/{filename}')

producer = KafkaProducer(bootstrap_servers=['35.189.215.83:9092'],
                         value_serializer=lambda x:
                         dumps(x).encode('utf-8'))

consumer = KafkaConsumer(
    'neuralnetwork_data',
    bootstrap_servers=['35.189.215.83:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    partition_assignment_strategy=[RoundRobinPartitionAssignor],
    group_id='NeuralNetworkService',
    value_deserializer=lambda x: loads(x.decode('utf-8')))


def predict(x):
    pred = net.predict(x)
    label = np.argmax(pred, axis=1)
    return label


def KafkaConsumer():
    while True:
        for message in consumer:
            tasks = np.asarray(message.value["Array"])
            result = np.reshape(tasks, (-1, 50, 7))
            for element in result:
                element[:] = element - element.mean(axis=0)
            pred = predict(result)[0]+1
            print(f'Partition: {message.partition}, Predict: {pred}')
            data = {'result': int(pred), 'label': message.value["Label"], 'id': message.value['Id']}
            producer.send('client_service_data', value=data)


kafkaConsumer = Thread(target=KafkaConsumer)
kafkaConsumer.start()
