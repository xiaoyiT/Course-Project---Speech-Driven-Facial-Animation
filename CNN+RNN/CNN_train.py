import tensorflow as tf
import numpy as np
from CNN import cnn_model_fn 

def main(unused_argv):

    tf.logging.set_verbosity(tf.logging.INFO)
    landmarks= np.load('./train/train_landmarks.npy')
    fft =  np.load('./train/train_fft.npy')
    train_data = fft  
    train_labels = landmarks

    speech_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./CNN_record")
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":train_data},y=train_labels,batch_size=300,num_epochs=5,shuffle=False)
    speech_classifier.train(input_fn=train_input_fn,steps=20000,hooks=[logging_hook])


if __name__ == "__main__":
    tf.app.run()
