# GAN_AudioGenerator
Repositorio de códigos versión beta de red GAN aplicada a la generación procedural de gestos sonoros electroacústicos.

El archivo GAN_AudioGenerator.py se entrena específicamente con audios en crudo y, posteriormente, se extrae el espectrograma para usar la matriz de valores de todos los sonidos del dataset para el entrenamiento de la GAN, dando como resultado un archivo .h5 en modo de inferencia para la generación de sonidos.

El archivo GAN_SpectrogramGenerator.py se entrena específicamente con imágenes de espectrogramas como datos de entrada y, por lo tanto, genera como resultado del entrenamiento de la red GAN imágenes de espectrogramas.


Visitar:
https://colab.research.google.com/drive/1gMpWnwGAEddZNFv6sMBH529x95AMny0i#scrollTo=yJHFytcJ8GTI
