# AppliedDeepLearning
Repository for Applied Deep Learning Course at TU Wien


## Topic: Computer Vision - Detection of Object - Pedestrian Detection

Nowadays, Computer Vision has taken a huge step forward due to major advancements in the field of Deep Learning. Having experienced object detection applications or heard about them in one way or another in my daily life, whether it's face recognition on my phone to reading articles about self driving cars, I have always found this topic very interesting. Therefore, I decided to concentrate on the task of pedestrian detection, a specialized area within the very broad field of object detection for my project.


## Reference Papers

A very interesting paper I read was the one about the F2DNET model that uses two-stage detectors which is state-of-the-art not only in object detection but also in pedestrian detection. Different from other methods, which at first employ a "weak" region detection method and later a "strong" one which refines the preselected regions,F2DNET actually does the opposite. It initally detects objects with a strong detection head and then refine and filter those detections with a lighter/ "weaker" second step. This leads to potentially faster and more accurate object detection and, therefore, F2DNET has achieved quite good results and has outperformed other very successful models. 

Despite achieving good performance through different models the main problem for pedestrian detection tasks remains the generalization issue. As represented in F2DNET but also Pedestrian which conducts a comprehensive study about this issue, performing cross-dataset evaluation can actually alleviate the problem. Also, in Pedestron the authors find out that using cross-dataset evaluation, object detectors without pedestrian tailored adaptions generalize quite better, which is something that I may also incorporate in my project.


## My Approach



### Dataset

### Method


### Work - Breakdown












## References

[F2DNET](https://arxiv.org/pdf/2203.02331v2.pdf)

@article{khan2022f2dnet,
  title={F2DNet: fast focal detection network for pedestrian detection},
  author={Khan, Abdul Hannan and Munir, Mohsin and van Elst, Ludger and Dengel, Andreas},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  pages={4658--4664},
  year={2022},
  organization={IEEE}
}

[PEDESTRON](https://arxiv.org/pdf/2003.08799v7.pdf)

@article{hasan2022pedestrian,
  title={Pedestrian Detection: Domain Generalization, CNNs, Transformers and Beyond},
  author={Hasan, Irtiza and Liao, Shengcai and Li, Jinpeng and Akram, Saad Ullah and Shao, Ling},
  journal={arXiv preprint arXiv:2201.03176},
  year={2022}
}
















