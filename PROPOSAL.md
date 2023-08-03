## Project proposal form

Please provide the information requested in the following form. Try provide concise and informative answers.

**1. What is your project title?**

Image Captioning Through Various Deep Learning Architectures

**2. What is the problem that you want to solve?**

Image Captioning is a process to generate a description given an image. We want to train models that could automatically gernerate a sentence describing the content of an image. This task lies at the intersection of Computer Vision and Nature Language Processing.

**3. What deep learning methodologies do you plan to use in your project?**

We are going to apply the following deep learning architectures to complete Image Captioning task:

1) CNN-LSTM Based Model: This model is implemented by [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555). Its architecture in this paper is LSTM model combined with a CNN image embedder and word embeddings.

2) CNN-LSTM-Attention Based Model: This model is implemented by [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). It injects Attention module into the CNN-LSTM based model, in order to automatically learns to describe the content of images.

3) RCNN-LSTM-Attention Based Model: This model is implemented by [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998). It combines bottom-up and top-down attention mechanism that enables attention to be calculated at the level of objects and other salient image regions. The bottom-up mechanism is based on Faster R-CNN.

4) Transformer Based Model: This model is implemented by [Attention on Attention for Image Captioning](https://arxiv.org/abs/1908.06954). The authors propose an Attention on Attention (AoA) module, which extends the conventional attention mechanisms to determine the relevance between attention results and queries. AoA first generates an information vector and an attention gate using the attention result and the current context, then adds another attention by applying element-wise multiplication to them and finally obtains the attended information, the expected useful knowledge.

We plan to rewrite the codes of the four model architectures listed above in Tensorflow or Pytorch, and then compare their performance with detailed explanation.

**4. What dataset will you use? Provide information about the dataset, and a URL for the dataset if available. Briefly discuss suitability of the dataset for your problem.**

We plan to use Common Objects in Context (COCO) dataset to complete this task.

COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image, and 250,000 people with keypoints. Here is the URL https://cocodataset.org/#home.

The data format of COCO designed for Image Captioning task is shown below. Each caption describes the specified image and each image has at least 5 captions (some images have more). Therefore, COCO is a very suitable dataset for Image Captioning task.

```
annotation{
	"id" : int, 
	"image_id" : int, 
	"caption" : str,
}
```

**5. List key references (e.g. research papers) that your project will be based on?**

O. Vinyals, A. Toshev, S. Bengio and D. Erhan, "Show and tell: A neural image caption generator," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, USA, 2015, pp. 3156-3164, doi: 10.1109/CVPR.2015.7298935.

Xu, K., Ba, J. L., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R. S., & Bengio, Y. (2015). Show, attend and tell: Neural image caption generation with visual attention. In F. Bach, & D. Blei (Eds.), 32nd International Conference on Machine Learning, ICML 2015 (pp. 2048-2057). (32nd International Conference on Machine Learning, ICML 2015; Vol. 3). International Machine Learning Society (IMLS).

Anderson, Peter & He, Xiaodong & Buehler, Chris & Teney, Damien & Johnson, Mark & Gould, Stephen & Zhang, Lei. (2018). Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering. 6077-6086. 10.1109/CVPR.2018.00636. 

L. Huang, W. Wang, J. Chen and X. -Y. Wei, "Attention on Attention for Image Captioning," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), Seoul, Korea (South), 2019, pp. 4633-4642, doi: 10.1109/ICCV.2019.00473.

**Please indicate whether your project proposal is ready for review (Yes/No):**

Yes

## Feedback (to be provided by the course lecturer)

[MV 1st April 2023] Project topic approved. Note that you must use TensorFlow in your project as this is what we use in the course. It is good that you identified a good set of references and that you plan to implement neural network architectures proposed in these papers. You may also want to implement and evaluate some original neural network architectures of your own and compare with baseline methods from the cited papers. I expect that your report will well explain the principles and rationale behind any neural network architectures you consider.  
