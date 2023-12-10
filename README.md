# AppliedDeepLearning
Repository for Applied Deep Learning Course at TU Wien


## Topic: Computer Vision 

### Assignment I

#### Image Caption Generator
Nowadays, Computer Vision and Natural Language Processing have taken a huge step forward due to major advancements in the field of Deep Learning and large volumes of data available. Different types of tasks like image classification, object detection, caption generator, sentiment analysis etc., which before were thought impossible, can now be solved by smart models and computers. 
Having always been fascinated by these two branches, I decided to concentrate for this project on the topic of `Image Captioning`, which actually lies at the intersection of Computer Vision and Natural Language Processing.


#### Related Work

[Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) by Kelvin Xu et al. which was published back on 2015 served as a good introduction to this topic. It suggests a CNN-RNN(Convolutional Neural Network - Recurrent Neural Network) model for generating image captions for Flickr30k and COCO datasets. Another interesting paper I read was [VLP](https://arxiv.org/pdf/1909.11059v3.pdf) by Zhou Luowei et al. which proposes a unified decoder-encoder model that uses a model pre-trained  on another dataset, fine tunes it, and then uses it to generate captions for COCO and Flickr30k datasets.


#### Approach

According to my research CNN-RNN model combinations are used quite often for this kind of task. Therefore, I will initially try to build a similar model for my project.

#### Dataset
At first I wanted to use the COCO dataset for my project but then I decided against it due to its large volume of over 1 million and half captions describing over 330000 images. Even though, I am going to use Google Colab Pro Version for this project, computation might still be a problem when training neural networks. 
Therefore, I decided to use the Flickr30k dataset. It contains around 31000 images collected from Flickr, together with 5 reference sentences provided by human annotators.

#### Prediction of Work-Breakdown Structure 

- Dataset Collection: I will use an already available dataset.
- Design and build of a model: I believe this will be the most challenging part of the project. It will require a lot of try and errors from my side and I think it might take up to 2 weeks.
- Train of model: As I plan to use GPU of Google Colab I hope it won't take more than 3 days, considering that my model might not run on the first try.
- Fine Tuning: This might also take me up to 1 week, as I have to do a lot of research and consider methods to improve my model.
- Application: This might take up to 1 week. I will probably build a website where I can upload a picture and it will give as output a caption describing it.
- Report and Presentation: 2 days.


### Assignment II

As mentioned above, [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)  was one of the main papers that served as a gate for me to the world of image captioning and I relied on it for the delivery of this project. Since it was my first time working with RNN and Attention Based models, my code was therefore heavily inspired by this paper but also by this Pytorch tutorial [a-PyTorch-Tutorial-to-Image-Captioning
](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning). Unfortunately, this was not state-of-art but it was still pretty good. 

### Error Metric
For reporting results of my implementation I used the so called BLEU metric which is actually a standard in the image captioning generator architecture. The table below shows the results of the implementations of the two previously mentioned papers:
| Paper  | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4|
| ---  | ------| ------| ------|------|
| Show, Attend and Tell | 66.7 | 43.4| 28.8 | 19.1 
| VLP |-| -| -| 31.1










### References

```
@article{Show, Attend and Tell,
  title={Show, Attend and Tell: Neural Image Caption
Generation with Visual Attention},
  author={Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio},
  journal={arXiv preprint arXiv:1502.03044 },
  year={2015}
}

@article{zhou2019vlp,
  title={Unified Vision-Language Pre-Training for Image Captioning and VQA},
  author={Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, Jianfeng Gao},
  journal={arXiv preprint arXiv:1909.11059},
  year={2019}
}
```





