import numpy as np

from network import create_model

from flask import Flask, jsonify, request

app = Flask(__name__)

filename = 'weights-ep2076-val_loss0.2518.hdf5'

net = create_model(training=False)

net.load_weights(f'../ckpt/{filename}')


def predict(x):
    pred = net.predict(x)
    label = np.argmax(pred, axis=1)  # [0]
    return label

@app.route('/predict', methods=['POST'])
def create_task():
    if request.data:
        tasks = []
        for j in request.get_json():
            task = [float(j['Gfx']), float(j['Gfy']), float(j['Gfz']),
                    float(j['Wx']), float(j['Wy']), float(j['Speed']),
                    float(j['Wz'])]
            tasks.append(task)
        result = np.reshape(tasks, (-1, 50, 7))
        for element in result:
            element[:] = element - element.mean(axis=0)
        something = predict(result)
    return jsonify(result=str(something[0]+1)), 200


if __name__ == '__main__':
    app.run(debug=True)
