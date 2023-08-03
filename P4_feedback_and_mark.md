**Project group 4: Image captioning through various deep learning architectures**

**Candidate numbers: 48119, 54751, 56111** 


In the report “Image captioning through various deep learning architectures”, applied different encoder-decoder neural network models for an image captioning task, using two Flickr datasets. This includes four models from existing literature and two alternations of a previously known model, namely Top-Down model by Anderson et al (2018).

The introduction clearly states the problem and what models are considered, making it clear what models are previously known and what models are new (extensions of previously known models). The introduction does not summarise the main findings of the evaluation. 

The related work section covers some key works on encoder-decoder neural network architectures. This section could have been more focused on references for the models considered in the work and positioning of the work in the report with these works. The meaning of some statements are not clear, e.g. “exploit the object relationships in the images with graph theory techniques”. Some concepts are referred to without a definition, e.g. REINFORCE algorithm and CIDEr. 

The methodology is covered in Section 3, Model Architecture. This section describes basic concepts of different models studied, namely, primary model (Section 3.1), Bahdanau model (Section 3.2), Top-down model (Section 3.3), AoA model (Section 3.4), and extensions (Section 3.5). Some arguments are not clear, e.g. saying that the architecture of Primary model is not enough to take full advantage of visual and language information, without explanation. Overall, main properties of the models are summarised well. The extensions consist of (1) replacing LSTM cells with GRU cells in the Top-Down model and (2) combining the refiner in AoA and the decoder in Top-Down model. The first extension is basic. The second extension could have been explained in some more details in Section 3.5. 

Training methods are described in Section 4, including hyperparameter setting (Section 4.1), training objective (Section 4.2), teacher forcing (Section 4.3), search strategies for generation of an output sequence (Section 4.4). The cross-entropy loss function in Equation (26) lacks a proper definition of t which is implicitly assumed to be such that t_i = 1 if true class is and t_i = 0 otherwise. This section should have defined the optimisation method used and the setting of parameters. 

The datasets used in the study, namely, Flickr8k and Flickr30k, are described in Section 5.1. 

The code consists of python files implementing different models, and some python code and scripts for tokenization and evaluation. The python code is well structured and commented. However, the code should have given credit for any original sources used. For example, the code for BahdanauAttention seems to have been inspired or taken from a source, e.g. similar code can be found here https://github.com/Lornatang/TensorFlow2-tutorials/blob/master/Experts_tutorial/Text/image_captioning.py. 

The numerical results are shown in Section 5, Experiment. Training loss versus the number of epochs is shown for different models, in Figure 7 for Flickr8k dataset and Figure 8, for Flickr30k dataset. Overall, these results indicate converge. The number of epochs is limited to 40. The training loss would further decrease by increasing the number of epochs. The report does not present the test loss versus the number of epochs which is a shortcoming. Showing this is standard and allows for checking for existence of overfitting. The evaluation metrics, namely BLEU and METEOR, are defined in Section 5.3. References should have been provided for these metrics, as well as some discussion about why using these particular metrics. The definitions would have been clearer by using mathematical formulation. The evaluation results are shown in Table 1 and Table 2 for the two datasets considered. AoA model consistently outperformed all other models. Overall, the performance of other models, except Primary model, and Top-Down refiner, is competitive to the best model. Some more discussion could have been provided about why Top-Down refiner performs no so well, as this is an extension model proposed in the report. The evaluation metrics considered allow for comparing different models against each other. Some discussion for absolute values of achieved BLEU and METEOR values could have been provided, and compared with those obtained in literature for related tasks. 

The conclusion section summarises the results of the report. Some more discussion for the proposed extensions could have been provided. Future work mentions a model based on vision transformers that could be considered as well as a larger MSCOCO dataset (not considered in the present study due to computational resources limits). 

The presentation quality of the report is overall good. The layout is well structured. The writing is overall clear. For improvement, all figures should have been referenced in the text. 

Overall this is a good report. Several models are considered and some extensions. Some more focused could have been provide on proposed extensions, especially trying to explain why one of the extensions did not achieve competitive performance. 

**Total mark: 68**

