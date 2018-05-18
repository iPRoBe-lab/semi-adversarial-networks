Model Files
==========

Trained Auxiliary Classifiers

  * Auxiliary gender predictor (`aux-gpred.pkl`)   
  * Auxiliary Face Matcher (download `vggface.pt-adj-255.pkl` from Google Drive [here (554MB)](https://drive.google.com/drive/folders/191levCYjD2U_lL93QJFy_KwSh0jIaMB7?usp=sharing), and the gray correction part `vgg_gray_corr_ptch.npz` to account for gray-scale input images)


Trained SAN model:
  * Convolutional Autoencoder after 20 epochs (`conv-autoencoder-e20.pkl`)   
   **Note:** when evaluating the SAN model, the auxiliary subnetworks are discarded.
