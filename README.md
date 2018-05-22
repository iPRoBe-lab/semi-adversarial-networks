Semi Adversarial Networks (SAN)
===============================

PyTorch implementation of the semi-adversarial neural network model described in *Semi-adversarial networks: Convolutional autoencoders for imparting privacy to face images* by V. Mirjalili, S. Raschka, A. Namboodiri, and A. Ross.

## Research Article

[Vahid Mirjalili](https://github.com/vmirly), [Sebastian Raschka](https://github.com/rasbt), [Anoop Namboodiri](https://www.iiit.ac.in/people/faculty/anoop/), and [Arun Ross](http://www.cse.msu.edu/~rossarun/) (2018) *Semi-adversarial networks: Convolutional autoencoders for imparting privacy to face images.* Proc. of 11th IAPR International Conference on Biometrics (ICB 2018), Gold Coast, Australia.   

- Arxiv preprint: [https://arxiv.org/abs/1712.00321](https://arxiv.org/abs/1712.00321)

## Implementation details and requirements


The model was implemented in PyTorch 0.3.1 using Python 3.6 and may be compatible with different versions of PyTorch and Python, but it has not been tested.

Additional requirements are listed in the [./requirements.txt](./requirements.txt) file. 


## Usage

**Source code and model parameters**

The source code of the SAN model can be found in the [src](./src) subdirectory, and the parameters of the trained models are available in the [model](./model) subdirectory.

**Dataset**

The original dataset was obtained from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, but the images in the dataset were cropped using DPM model (source code can be found in http://www.robots.ox.ac.uk/~vgg/software/vgg_face/). In addition, we have aligned the prototypes to each input image.

We made these preprocessed images available via the following link. In order to run these
https://drive.google.com/drive/folders/191levCYjD2U_lL93QJFy_KwSh0jIaMB7?usp=sharing

**Training a SAN model**

In order to train a SAN model as described in the research paper, 
download the preprocessed image files (see previous paragraph) and place them into the main directory of this GitHub repository. To initiate the model training, execute the following code in your shell terminal.

```bash
cd src/
python main.py --gpu
```

Note that the `--gpu` flag specifies which GPU on your system shall be used; `--gpu 0` uses the first CUDA GPU device, `--gpu 1` uses the second CUDA GPU device and so forth. 



**Examples**

Please see the [examples/evaluate-SAN-model.ipynb](examples/evaluate-SAN-model.ipynb) Jupyter Notebook for an example that illustrates how to instantiate the trained autoencoder part of the SAN model for evaluation, using the available model parameters. 


More usage examples may follow in future.