<<<<<<< HEAD
<!-- Overall Aim:

To develop and evaluate a computationally efficient, real-time continuous sign language recognition system that operates on consumer-grade hardware while maintaining sufficient accuracy (> 85%) for practical deployment in educational settings, thereby improving accessibility for deaf and hard-of-hearing students in digital learning environments.

Objectives:

Design and implement an efficient feature extraction pipeline
Develop a hybrid feature extraction framework combining MediaPipe Holistic landmarks (i.e., pose, hand, and face keypoints) with lightweight CNN features to create compact yet discriminative representations of sign language gestures, reducing computational overhead by ~70% compared to traditional video-based approaches while maintaining temporal coherence.
Create a memory-efficient temporal modelling architecture
Design and implement a novel temporal fusion architecture combining MobileNetV3 with bidirectional LSTM and selective attention mechanisms, efficient to operate within 8GB VRAM constraints through gradient checkpointing, mixed-precision training and dynamic batching strategies, achieving a reasonable model size under 100MB.
	Develop a knowledge distillation framework for model compression
Implement a multi-stage knowledge distillation pipeline transferring knowledge from large pre-trained models (e.g., I3D, SlowFast) to our lightweight architecture, utilizing both soft targets and intermediate feature matching to maintain 95% of teacher model accuracy while reducing parameters by 90%.
	Evaluate system performance in real-world educational context
Conduct a comprehensive evaluation including: (a) quantitative assessment on RWTH-PHOENIX-Weather 2014 dataset achieving <25% Word Error Rate, (b) real-time performance validation achieving 30+ FPS inference, (c) user study with 5-10 deaf/hard-of-hearing students measuring communication effectiveness and system usability, and (d) cross-domain adaptation to educational vocabulary.

	Methodology:

	Data Collection and Processing
The research will utilize the RWTH-PHOENIX-Weather 2014 dataset containing ~8,257 annotated sequences of German Sign Language with over 1,000-sign vocabulary. The dataset’s modest size (53GB) and low resolution (210x260px) make it ideal for resource-constrained training. Videos will be pre-processed using MediaPipe Holistic to extract over 500 spatial landmarks (33 pose + 42 hand + 468 face keypoints), creating a structured representation that reduces memory requirements by ~80% compared to raw video processing.
	Model Development Approach:
The methodology follows an iterative development cycle with three main phases:
Phase I – Baseline Development: Implement a CNN-HMM baseline following Koller at al. (2015) to establish performance benchmarks. Extract and cache MediaPipe features offline to optimize training efficiency. This phase will validate the technical feasibility and establish baseline metrics (expected at least 40% WER)
Phase II – Architecture Optimization: Develop the proposed lightweight architecture combining MobileNetV3 backbone (Howard et al., 2019) with BiLSTM temporal modelling. Implement memory efficient techniques including:
	Mixed-precision (FP16) training reducing memory usage by 50%.
	Gradient checkpointing saving 70% activation memory.
	Dynamic sequence truncation based on available memory.
	Feature quantization reducing storage requirements.
Apply knowledge distillation using temperature-scaled soft targets (T=3.0) from pre-trained I3D teacher models, optimizing the loss function:
L=0.7*L_soft+0.3*L_hard
Phase III – Real-time Deployment: Optimize inference through TensorRT compilation achieving 2-3x speedup, implement sliding window processing with 32-frame buffers and 8-frame stride for continuous recognition, and develop web-based deployment using TensorFlow.js for browser compatibility.
	Evaluation Framework
Evaluation will employ multiple metrics across different dimensions:
Recognition Performance: Word Error Rate (WER) as primary metric, Sign Error Rate (SER) for isolated signs, and BLEU scores for translation quality assessment.
Computational Efficiency: Real-time factor (RTF), end-to-end latency measurements, memory footprint profiling, and frames-per-second (FPS) throughput.
User Experience: System Usability Scale (SUS) questionnaire, task completion rates for educational scenarios, and qualitative feedback through semi-structured interviews.
	Experimental Design
Experiments will follow a controlled ablation study design, systematically evaluating contributions of each component
	Baseline CNN-HMM vs. proposed architecture
	With/without knowledge distillation
	Impact of different temporal window size (16, 32, 64 frames)
	Effect of attention mechanisms
	Comparison of optimization techniques
All experiments will be conducted with fixed random seeds for reproducibility, using 5-folds cross-validation on the training set, with results reported as mean ±standard deviation.
	Structure
Chapter I: Introduction 
	Problem statement and motivation
	Research questions and hypotheses
	Contributions and thesis outline
	Societal impact and accessibility considerations
Chapter II: Literature Review
	Evolution of sign language recognition systems
	Deep learning approaches for video understanding
	Model compression and knowledge distillation techniques
	Real-time deployment strategies
	Gap analysis and research opportunities
Chapter III: Theoretical Framework
	Sign language linguistics and phonology
	Spatial-temporal feature representation
	Continuous sequence modelling with Connectionist Temporal Classification (CTC)
	Knowledge distillation theory
	Optimization techniques for edge deployment
Chapter IV: Methodology
	Dataset description and preprocessing pipeline
	Proposed architecture design
	Training strategies and optimization
	Evaluation metrics and protocols
	Implementation details
Chapter V: Experimental Results
	Baseline experiments and analysis
	Ablation studies on architectural components
	Knowledge distillation effectiveness
	Real-time performance evaluation
	Comparison with state-of-the-art methods
Chapter VI: User Study and Educational Application
	Study design and participant recruitment
	Educational vocabulary adaptation
	Usability evaluation results
	Qualitative feedback analysis
	Deployment case studies
Chapter VII: Discussion
	Interpretation of results
	Limitation and challenges
	Theoretical implications
	Practical considerations for deployment
	Ethical considerations and bias analysis
Chapter VIII: Conclusion and Future Work
	Summary of contributions
	Achievement of objectives
	Future research directions
	Broader impact on accessibility

	Preliminary Reading List
Foundational Sign Language Recognition
	Koller, O., Forster, J., & Ney, H. (2015). Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. Computer Vision and Image Understanding, 141, 108-125.
	Camgöz, N. C., Hadfield, S., Koller, O., Ney, H., & Bowden, R. (2018). Neural sign language translation. IEEE Conference on Computer Vision and Pattern Recognition, 7784-7793.
	Cui, R., Liu, H., & Zhang, C. (2019). A deep neural framework for continuous sign language recognition by iterative training. IEEE Transactions on Multimedia, 21(7), 1880-1891.
Efficient Deep Learning Architectures
	Howard, A., Sandler, M., et al. (2019). Searching for MobileNetV3. Proceedings of the IEEE/CVF International Conference on Computer Vision, 1314-1324.
	Zhang, Y., et al. (2024). VideoMamba: State space model for efficient video understanding. European Conference on Computer Vision (ECCV).
	Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. International Conference on Machine Learning, 6105-6114.
Knowledge Distillation and Model Compression
	Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
	Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations.
	Jiao, X., et al. (2020). TinyBERT: Distilling BERT for natural language understanding. Findings of EMNLP, 4163-4174.
Real-time Deployment and Optimization
	Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172.
	Nikitin, N., & Fomin, E. (2025). Developing lightweight DNN models with limited data for real-time sign language recognition. arXiv preprint arXiv:2507.00248.
	Li, J., et al. (2024). Neural video compression with feature modulation. IEEE/CVF Conference on Computer Vision and Pattern Recognition.
Temporal Modeling and Sequence Learning
	Graves, A., et al. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. ICML, 369-376.
	Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.
	Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.
Educational Technology and Accessibility
	Bragg, D., et al. (2019). Sign language recognition, generation, and translation: An interdisciplinary perspective. ASSETS, 16-31.
	Baihan, A., et al. (2024). Sign language recognition using modified deep learning network and hybrid optimization. Scientific Reports, 14, 26111.
	Zhang, J., et al. (2024). Sign language recognition based on dual-path background erasure convolutional neural network. Scientific Reports, 14(1), 11360.
Datasets and Benchmarks
	Mukushev, M., et al. (2022). FluentSigners-50: A signer independent benchmark dataset for sign language processing. PLoS ONE, 17(9).
	Forster, J., et al. (2012). RWTH-PHOENIX-Weather: A large vocabulary sign language recognition and translation corpus. LREC, 3785-3789.
 -->
=======
	<!-- Overall Aim:

To develop and evaluate a computationally efficient, real-time continuous sign language recognition system that operates on consumer-grade hardware while maintaining sufficient accuracy (> 85%) for practical deployment in educational settings, thereby improving accessibility for deaf and hard-of-hearing students in digital learning environments.

	Objectives:

	Design and implement an efficient feature extraction pipeline
Develop a hybrid feature extraction framework combining MediaPipe Holistic landmarks (i.e., pose, hand, and face keypoints) with lightweight CNN features to create compact yet discriminative representations of sign language gestures, reducing computational overhead by ~70% compared to traditional video-based approaches while maintaining temporal coherence.
	Create a memory-efficient temporal modelling architecture
Design and implement a novel temporal fusion architecture combining MobileNetV3 with bidirectional LSTM and selective attention mechanisms, efficient to operate within 8GB VRAM constraints through gradient checkpointing, mixed-precision training and dynamic batching strategies, achieving a reasonable model size under 100MB.
	Develop a knowledge distillation framework for model compression
Implement a multi-stage knowledge distillation pipeline transferring knowledge from large pre-trained models (e.g., I3D, SlowFast) to our lightweight architecture, utilizing both soft targets and intermediate feature matching to maintain 95% of teacher model accuracy while reducing parameters by 90%.
	Evaluate system performance in real-world educational context
Conduct a comprehensive evaluation including: (a) quantitative assessment on RWTH-PHOENIX-Weather 2014 dataset achieving <25% Word Error Rate, (b) real-time performance validation achieving 30+ FPS inference, (c) user study with 5-10 deaf/hard-of-hearing students measuring communication effectiveness and system usability, and (d) cross-domain adaptation to educational vocabulary.

	Methodology:

	Data Collection and Processing
The research will utilize the RWTH-PHOENIX-Weather 2014 dataset containing ~8,257 annotated sequences of German Sign Language with over 1,000-sign vocabulary. The dataset’s modest size (53GB) and low resolution (210x260px) make it ideal for resource-constrained training. Videos will be pre-processed using MediaPipe Holistic to extract over 500 spatial landmarks (33 pose + 42 hand + 468 face keypoints), creating a structured representation that reduces memory requirements by ~80% compared to raw video processing.
	Model Development Approach:
The methodology follows an iterative development cycle with three main phases:
Phase I – Baseline Development: Implement a CNN-HMM baseline following Koller at al. (2015) to establish performance benchmarks. Extract and cache MediaPipe features offline to optimize training efficiency. This phase will validate the technical feasibility and establish baseline metrics (expected at least 40% WER)
Phase II – Architecture Optimization: Develop the proposed lightweight architecture combining MobileNetV3 backbone (Howard et al., 2019) with BiLSTM temporal modelling. Implement memory efficient techniques including:
	Mixed-precision (FP16) training reducing memory usage by 50%.
	Gradient checkpointing saving 70% activation memory.
	Dynamic sequence truncation based on available memory.
	Feature quantization reducing storage requirements.
Apply knowledge distillation using temperature-scaled soft targets (T=3.0) from pre-trained I3D teacher models, optimizing the loss function:
L=0.7*L_soft+0.3*L_hard
Phase III – Real-time Deployment: Optimize inference through TensorRT compilation achieving 2-3x speedup, implement sliding window processing with 32-frame buffers and 8-frame stride for continuous recognition, and develop web-based deployment using TensorFlow.js for browser compatibility.
	Evaluation Framework
Evaluation will employ multiple metrics across different dimensions:
Recognition Performance: Word Error Rate (WER) as primary metric, Sign Error Rate (SER) for isolated signs, and BLEU scores for translation quality assessment.
Computational Efficiency: Real-time factor (RTF), end-to-end latency measurements, memory footprint profiling, and frames-per-second (FPS) throughput.
User Experience: System Usability Scale (SUS) questionnaire, task completion rates for educational scenarios, and qualitative feedback through semi-structured interviews.
	Experimental Design
Experiments will follow a controlled ablation study design, systematically evaluating contributions of each component
	Baseline CNN-HMM vs. proposed architecture
	With/without knowledge distillation
	Impact of different temporal window size (16, 32, 64 frames)
	Effect of attention mechanisms
	Comparison of optimization techniques
All experiments will be conducted with fixed random seeds for reproducibility, using 5-folds cross-validation on the training set, with results reported as mean ±standard deviation.
	Structure
Chapter I: Introduction 
	Problem statement and motivation
	Research questions and hypotheses
	Contributions and thesis outline
	Societal impact and accessibility considerations
Chapter II: Literature Review
	Evolution of sign language recognition systems
	Deep learning approaches for video understanding
	Model compression and knowledge distillation techniques
	Real-time deployment strategies
	Gap analysis and research opportunities
Chapter III: Theoretical Framework
	Sign language linguistics and phonology
	Spatial-temporal feature representation
	Continuous sequence modelling with Connectionist Temporal Classification (CTC)
	Knowledge distillation theory
	Optimization techniques for edge deployment
Chapter IV: Methodology
	Dataset description and preprocessing pipeline
	Proposed architecture design
	Training strategies and optimization
	Evaluation metrics and protocols
	Implementation details
Chapter V: Experimental Results
	Baseline experiments and analysis
	Ablation studies on architectural components
	Knowledge distillation effectiveness
	Real-time performance evaluation
	Comparison with state-of-the-art methods
Chapter VI: User Study and Educational Application
	Study design and participant recruitment
	Educational vocabulary adaptation
	Usability evaluation results
	Qualitative feedback analysis
	Deployment case studies
Chapter VII: Discussion
	Interpretation of results
	Limitation and challenges
	Theoretical implications
	Practical considerations for deployment
	Ethical considerations and bias analysis
Chapter VIII: Conclusion and Future Work
	Summary of contributions
	Achievement of objectives
	Future research directions
	Broader impact on accessibility

	Preliminary Reading List
Foundational Sign Language Recognition
	Koller, O., Forster, J., & Ney, H. (2015). Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. Computer Vision and Image Understanding, 141, 108-125.
	Camgöz, N. C., Hadfield, S., Koller, O., Ney, H., & Bowden, R. (2018). Neural sign language translation. IEEE Conference on Computer Vision and Pattern Recognition, 7784-7793.
	Cui, R., Liu, H., & Zhang, C. (2019). A deep neural framework for continuous sign language recognition by iterative training. IEEE Transactions on Multimedia, 21(7), 1880-1891.
Efficient Deep Learning Architectures
	Howard, A., Sandler, M., et al. (2019). Searching for MobileNetV3. Proceedings of the IEEE/CVF International Conference on Computer Vision, 1314-1324.
	Zhang, Y., et al. (2024). VideoMamba: State space model for efficient video understanding. European Conference on Computer Vision (ECCV).
	Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. International Conference on Machine Learning, 6105-6114.
Knowledge Distillation and Model Compression
	Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
	Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations.
	Jiao, X., et al. (2020). TinyBERT: Distilling BERT for natural language understanding. Findings of EMNLP, 4163-4174.
Real-time Deployment and Optimization
	Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172.
	Nikitin, N., & Fomin, E. (2025). Developing lightweight DNN models with limited data for real-time sign language recognition. arXiv preprint arXiv:2507.00248.
	Li, J., et al. (2024). Neural video compression with feature modulation. IEEE/CVF Conference on Computer Vision and Pattern Recognition.
Temporal Modeling and Sequence Learning
	Graves, A., et al. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. ICML, 369-376.
	Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.
	Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.
Educational Technology and Accessibility
	Bragg, D., et al. (2019). Sign language recognition, generation, and translation: An interdisciplinary perspective. ASSETS, 16-31.
	Baihan, A., et al. (2024). Sign language recognition using modified deep learning network and hybrid optimization. Scientific Reports, 14, 26111.
	Zhang, J., et al. (2024). Sign language recognition based on dual-path background erasure convolutional neural network. Scientific Reports, 14(1), 11360.
Datasets and Benchmarks
	Mukushev, M., et al. (2022). FluentSigners-50: A signer independent benchmark dataset for sign language processing. PLoS ONE, 17(9).
	Forster, J., et al. (2012). RWTH-PHOENIX-Weather: A large vocabulary sign language recognition and translation corpus. LREC, 3785-3789.
 -->
>>>>>>> b674efc90889da5b013e835083ef6c665f57ef6d
