1. Download video from RAVDESS database.
2. Put videos under folder /video.
3. mkdir data/fr.
4. Run "python prep_gan.py" to extract facial images from videos, the output images will be saved in /data/fr/ folder and landmarks will be saved in ./data/landmark.npy.
5. Run "python main.py --train --epoch 25" to train GAN network.
6. Run "python main.py" to generate sample images.