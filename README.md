# AppliedDeepLearning
Repository for Applied Deep Learning Course at TU Wien


## Topic: Computer Vision  
### Image Caption Generator

Nowadays, Computer Vision and Natural Language Processing have taken a huge step forward due to major advancements in the field of Deep Learning and large volumes of data available. Different types of tasks like image classification, object detection, caption generator, sentiment analysis etc., which before were thought impossible, can now be solved by smart models and computers. 
Having always been fascinated by these two branches, I decided to concentrate for this project on the topic of `Image Captioning`, which actually lies at the intersection of computer vision and natural language processing.


### Related Work

[Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) which was published back on 2015 served as a good introduction to this topic. It suggests a CNN-RNN(Convolutional Neural Network - Recurrent Neural Network) model for generating image captions for Flickr30k and COCO datasets.

Another interesting paper I read was [VLP](https://arxiv.org/pdf/1909.11059v3.pdf) which describes a technique that uses a pre-trained model on another dataset which is then later fine-tuned for vision-language generation on COCO and Flickr30k datasets.


### Approach

According to my research CNN-RNN combinations are used quite often for this kind of task. Therefore, I will initially try to build a similar model as described in the first paper. 


### Dataset
At first I wanted to use the COCO dataset for my project but then I decided against it and the only reason was its large volume of over 1 million and half captions describing over 330000 images. Even though, I am going to use Colab Pro Version for this project, computation might still be a problem when training neural networks. Therefore, I decided to user the Flickr30k dataset. It contains around 31000 images collected from Flickr, together with 5 reference sentences provided by human annotators.


### Work-Breakdown Structure

- Dataset Collection: I will use already available dataset
- Design and build of a model: I believe this will be the most challenging part of the project. It will require,in addition, a lot of data preparation and a lot of try and errors from my side. I think it might take up to 1 week.
- Train of model: As I plan to use GPU of Google Colab I hope it won't take more than 3 days, considering that my model might not run on the first try
- Fine Tuning: This might take also up to 1 week, as I have to do a lot of research and consider methods to improve my model
- Application: This might take up to 5 days. I will probably build a website where I can upload a picture and it will give as output a caption describing it
- Report and Presentation: 1 day


### References


[Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

@article{Show, Attend and Tell,
  title={Show, Attend and Tell: Neural Image Caption
Generation with Visual Attention},
  author={Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio},
  journal={arXiv preprint arXiv:1502.03044 },
  year={2015}
}

[VLP](https://arxiv.org/pdf/1909.11059v3.pdf)

@article{VLP,
  title={Unified Vision-Language Pre-Training for Image Captioning and VQA,
  author={Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, Jianfeng Gao},
  journal={arXiv preprint 	arXiv:1909.11059},
  year={2019}
}




