[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Super Resolution example</h3>

  <p align="center">
    SRCNN, U-Net, ESRGAN model implementation for SISR
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#prerequisites-and-install">Prerequisites and Install</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#todo">TODO</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![Product Name Screen Shot][esrgan-screenshot]

![Product Name Screen Shot][srcnn-screenshot]

Example implementation of several well known SISR architectures

Papers:
* [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)
* [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

Project focuses on SISR (Single Image Super Resolution) using different architectures. Goal of the project is to use it for video upscaling by single image (frame) at the beginning
and enhance it with some Temporal GAN implementation (e.g. iSeeBetter, FRVSR-GAN) later on.

Still in progress.

### Built With

* [PyTorch](https://pytorch.org)

### Prerequisites and Install

* Python3
* pip
  ```sh
  pip install -r requirements.txt
  ```

<!-- USAGE EXAMPLES -->
## Usage

###Dataset

Image patches from T91 dataset can be created by using utils/image.py script
  ```sh
  python utils/image.py
  ```

###Train

Train model by using
  ```sh
  python train.py
  ```

<!-- ROADMAP -->
## TODO:
- script for video upscaling frame-by-fram using SISR model
- Temporal GAN implementation (e.g. iSeeBetter, FRVSR-GAN)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

[LinkedIn](https://www.linkedin.com/in/jirislapnicka/)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jirislapnicka/
[esrgan-screenshot]: images/esrgan_preview.png
[srcnn-screenshot]: images/srcnn_preview.png