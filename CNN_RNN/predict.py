import numpy as np
from pydub import AudioSegment
from scipy.fftpack import fft
import tensorflow as tf
from CNN import cnn_model_fn 
import sys
import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import imageio
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):

    if(len(argv)<=0):
    	print("Please Input File Path")
    	sys.exit(2)

    print("Reading Speech File ...")
    file=""
    for arg in argv:
    	if(arg.endswith(".wav")):
    		file=arg;
    		break;

    try:
	    old_speech = AudioSegment.from_wav(file)
	    duration=len(old_speech)

	    all_fft=[]
	    step=30
	    cnt1=-1
	    cnt2=1
	    while cnt2 * step<=duration:
	        t1 = cnt1 * step 
	        t2 = cnt2 * step
	        if cnt1>=0:
	            newAudio = old_speech[t1:t2]
	        else:
	            newAudio= AudioSegment.silent(duration=step,frame_rate=old_speech.frame_rate)
	            newAudio=newAudio+old_speech[0:t2]
	        data = newAudio.get_array_of_samples()

	        normalize_data=[(ele/2**16.)*2-1 for ele in data] # this is 8-bit track, b is now normalized on [-1,1)
	        if len(normalize_data)!=0:
	            audio_fft=[]
	            fft_step=128;
	            fft_cnt=0;
	            fft_size=len(normalize_data)
	            result=[]
	            while (fft_cnt)*fft_step<fft_size and fft_cnt<23:
	                fft_reuslt = fft(x=normalize_data[fft_cnt*fft_step:(fft_cnt+1)*fft_step],n=fft_step) 
	                result.append(abs(fft_reuslt).tolist())
	                fft_cnt+=1
	            if all_fft:
	                while fft_cnt<23:
	                    result.append(np.zeros(fft_step).tolist())
	                    ft_cnt+=1
	            audio_fft.append(result)
	            all_fft.append(np.array(audio_fft).T.tolist())
	            audio_fft=[]
	        cnt1+=1
	        cnt2+=1
    except Error:
        print("No Such File.")
        sys.exit(2)

    print("Read Speech File Finished. Predicting ... ")
    speech_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./CNN_record")

    cnn_input=np.array(all_fft,dtype=np.float32)


    writer=imageio.get_writer('./landmarks.gif', mode='I',fps=24) 
    figure_cnt=1
    for speech_fft in cnn_input:
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":speech_fft},shuffle=False)
        predict_result= list(speech_classifier.predict(input_fn=predict_input_fn))
        landmarks=predict_result[0]["classes"].tolist()
        landmarks_x=[]
        landmarks_y=[]
        for cnt in range(0,136,2):
            landmarks_x.append(landmarks[cnt])
            landmarks_y.append(-landmarks[cnt+1])
        fig=plt.figure(figure_cnt)
        plt.plot(landmarks_x,landmarks_y,'o')
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')
        canvas.draw()
        ncols, nrows = canvas.get_width_height()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(nrows, ncols, 3)
        fig.savefig("./predict_landmark/predict"+str(figure_cnt)+".png")
        writer.append_data(image)
        figure_cnt+=1
        # plt.show()
    writer.close()
    print("Landmark Prediction Finished")


if __name__ == "__main__":
    main(sys.argv)