!!!!!Under construction!!!!

Zero shot image denoising based on downsampling
---
#### Sadjad Rezvani, x, y, z
#### Paper: [https://ieeexplore.ieee.org/iel8/6287639/6514899/10654252.pdf]
> In this work, we tackle the complex challenge of image denoising, especially in scenarios where high-resolution images are corrupted by noise, making it difficult to restore their quality. Our proposed Fast Residual Denoising Framework (FRDF) leverages zero-shot learning principles, allowing it to effectively denoise images without the need for clean/noisy image pairs for training. With only 23K parameters, our model is lightweight and efficient, making it suitable for real-world applications ranging from medical imaging to surveillance, even with limited computational resources.
### Network Architecture
![main](https://github.com/user-attachments/assets/d5960eaf-ba06-4967-8e56-6bb2c48c8220)


### News

### Installation
how to  install
NOT excited reqirement.txt 

```python
python 3.6.9
pytorch 1.5.1
```
```
git clone https://github.com/sadjadrz/Zero-shot-image-denoising-based-on-downsampling
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Quick Start (Single Image Inference)
---
* ```python basicsr/demo.py -opt options/demo/demo.yml```
  * modified your 

### Results

---
Some of the images.

### License

This project is under the MIT license.

### Citations

If Zero shot image denoising based on downsampling helps your research or work, please consider citing it.
```
@article{rezvani2024single,
  title={Single Image Denoising via a New Lightweight Learning-Based Model},
  author={Rezvani, Sadjad and Siahkar, Fatemeh Soleymani and Rezvani, Yasin and Gharahbagh, Abdorreza Alavi and Abolghasemi, Vahid},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```


### Contact
If you have any questions, please contact sadjadRezvani@gmail.com.

