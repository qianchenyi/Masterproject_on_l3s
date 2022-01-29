# Masterproject_on_l3s
The files in the PROJECT:
1. Image_Processing.py is used define the colormap and convert the malware dataset "MalImg" to grayscale images with colormap. Some functions about reading files are also defined here.

2. benign_processing_new.py is used to convert the collected benign executables to greyscale images with and without colomap.

3. Blackbox.py is about training the blackbox detector, and during the training of GAN model, this file is also used to assign labels for generated samples.

4. MasterGAN.py defined the GAN.

5. DataUploader is used when we train the blackbox detector.

6. we use main.py to start the training of the GAN.