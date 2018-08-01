# Prediction input file with learning data

import numpy as np
import tensorflow as tf

imagePath = 'tensorflow_box-master/test1.jpg'
modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'

answer = None

image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-5:][::-1]  # get predictions high top 5
    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\\n", "") for w in lines]
    for node_id in top_k:
        labels_name = labels[node_id]
        score = predictions[node_id]*100
        if 'pos' in labels_name:
            if score > 0.7:
                print('This image include a car!')
            else:
                print('This image does not include a car.')
            print('score : %.5f / 100'%score)