# Awesome GAN Papers



A curated list of GAN papers

- [GAN Basics with code (DCGAN, WGAN, Pix2Pix, CycleGAN, ProgressiveGAN, StyleGAN)](https://github.com/ysbsb/awesome-gan-papers#gan-basics)
- [GAN Metrics (Inception score, FID)](https://github.com/ysbsb/awesome-gan-papers#gan-metric)
- [GAN papers accepted on conferences (NIPS 2020, ICML 2020, CVPR 2020)](https://github.com/ysbsb/awesome-gan-papers#conference)

<br>



<h1>GAN Basics</h1>

Generative Adversarial Networks with codes

|   GAN model    |                            Tiitle                            |   conference   |                       tensorflow code                        |                         pytorch code                         |
| :------------: | :----------------------------------------------------------: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     DCGAN      | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) |   ICLR 2016    |                                                              | [pytorch example](https://github.com/pytorch/examples/tree/master/dcgan) |
|      WGAN      |     [Wasserstein GAN](https://arxiv.org/abs/1701.07875)      |                |                                                              | [official](https://github.com/martinarjovsky/WassersteinGAN) |
|    WGAN-GP     | [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) |   NIPS 2017    |      [official](https://github.com/openai/improved-gan)      |       [unofficial](https://github.com/caogang/wgan-gp)       |
|     BEGAN      | [BEGAN: Boundary Equilibrium Generative Adversarial Networks]() |   arXiv 2017   | [unofficial](https://github.com/carpedm20/BEGAN-tensorflow)  |   [unofficial](https://github.com/carpedm20/BEGAN-pytorch)   |
|    pix2pix     | [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) |   CVPR 2017    |                                                              |       [official](https://github.com/phillipi/pix2pix)        |
|    CycleGAN    | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) |   ICCV 2017    |                                                              |       [official](https://github.com/junyanz/CycleGAN)        |
|     GauGAN     | [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) | CVPR 2019 oral |                                                              |         [official](https://github.com/NVlabs/SPADE)          |
| ProgressiveGAN | [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) |   ICLR 2018    | [official](https://github.com/tkarras/progressive_growing_of_gans) | [unofficial1](https://github.com/nashory/pggan-pytorch), [unofficial2](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans) |
|    StyleGAN    | [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) |   CVPR 2019    |        [official](https://github.com/NVlabs/stylegan)        | [unofficial1](https://github.com/rosinality/style-based-gan-pytorch), [unofficial2](https://github.com/tomguluson92/StyleGAN_PyTorch) |



<br>

<h1>GAN Metrics</h1>



|     Metric      |                            Tiitle                            |     conference     |                  tensorflow code                   |                         pytorch code                         |
| :-------------: | :----------------------------------------------------------: | :----------------: | :------------------------------------------------: | :----------------------------------------------------------: |
| Inception Score | [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) |     NIPS 2017      | [official](https://github.com/openai/improved-gan) |                                                              |
| Inception Score | [A note on the Inception Score](https://arxiv.org/abs/1801.01973) | ICML 2018 Workshop |                                                    | [unofficial](https://github.com/sbarratt/inception-score-pytorch) |
|       FID       | [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium) |     NIPS 2017      |   [official](https://github.com/bioinf-jku/TTUR)   |                                                              |
|       FID       | [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium) |     NIPS 2017      |                                                    |    [unofficial](https://github.com/mseitzer/pytorch-fid)     |

<br>







<h1>Conference</h1>

<h2>ICLR 2021</h2>

Oral

- Do 2D GANs know 3D shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs
- Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering

Spotlight

- On Self-Supervised Image Representations for GAN Evaluation
- Influence Estimation for Generative Adversarial Networks
- Large Scale Image Completion via Co-Modulated Generative Adversarial Networks

Poster

- Private Post-GAN Boosting
- Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis
- GAN "Steerability" without optimization
- DINO: A Conditional Energy-Based GAN for Domain Translation
- GANs Can Play Lottery Tickets Too
- Training GANs with Stronger Augmentations via Contrastive Discriminator
- Enjoy Your Editing: Controllable GANs for Image Editing via Latent Space Navigation
- Using latent space regression to analyze and leverage compositionality in GANs
- GAN2GAN: Generative Noise Learning for Blind Denoising with Single Noisy Images
- Taming GANs with Lookahead-Minmax
- Wasserstein-2 Generative Networks

- Counterfactual Generative Networks
- Reducing the Computational Cost of Deep Generative Models with Binary Neural Networks
- Evaluating the Disentanglement of Deep Generative Models through Manifold Topology
- Adaptive and Generative Zero-Shot Learning
- CcGAN: Continuous Conditional Generative Adversarial Networks for Image Generation
- Understanding Over-parameterization in Generative Adversarial Networks





<h2>NIPS 2020</h2>



- ColdGANs: Taming Language GANs with Cautious Sampling Strategies  [paper](https://arxiv.org/abs/2006.04643)

- HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis [paper](https://arxiv.org/abs/2010.05646  https://github.com/jik876/hifi-gan)

- GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators  [paper](https://arxiv.org/abs/2006.08265)

- GANSpace: Discovering Interpretable GAN Controls [paper](https://arxiv.org/abs/2004.02546)  [code](https://github.com/harskish/ganspace)

- Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample [paper](https://arxiv.org/abs/2006.12226)

- Distributional Robustness with IPMs and links to Regularization and GANs  [paper](https://arxiv.org/abs/2006.04349)

- TaylorGAN: Neighbor-Augmented Policy Update for Sample-Efficient Natural Language Generation  [paper](https://arxiv.org/abs/2011.13527)

- DeepI2I: Enabling Deep Hierarchical Image-to-Image Translation by Transferring from GANs  [paper](https://arxiv.org/abs/2011.05867)

- GramGAN: Deep 3D Texture Synthesis From 2D Exemplars  [paper](https://arxiv.org/abs/2006.16112)

- Instance Selection for GANs  [paper](https://arxiv.org/abs/2007.15255)

- COT-GAN: Generating Sequential Data via Causal Optimal Transport  [paper](https://arxiv.org/abs/2006.08571)

- Elastic-InfoGAN: Unsupervised Disentangled Representation Learning in Class-Imbalanced Data  [paper](https://arxiv.org/abs/1910.01112)

- Improving GAN Training with Probability Ratio Clipping and Sample Reweighting  [paper](https://arxiv.org/abs/2006.06900)

- BlockGAN: Learning 3D Object-aware Scene Representations from Unlabelled Images  [paper](https://arxiv.org/abs/2002.08988)

- Teaching a GAN What Not to Learn  [paper](https://arxiv.org/abs/2010.15639)

- GAN Memory with No Forgetting  [paper](https://arxiv.org/abs/2006.07543)  [code](https://github.com/MiaoyunZhao/GANmemory_LifelongLearning)

- PlanGAN: Model-based Planning With Sparse Rewards and Multiple Goals  [paper](https://arxiv.org/abs/2006.00900)

- Your GAN is Secretly an Energy-based Model and You Should Use Discriminator Driven Latent Sampling  [paper](https://neurips.cc/Conferences/2020/Schedule?showEvent=17493)

- Towards a Better Global Loss Landscape of GANs  [paper](https://arxiv.org/abs/2011.04926)

- Differentiable Augmentation for Data-Efficient GAN Training  [paper](https://arxiv.org/abs/2006.10738) [code](https://github.com/mit-han-lab/data-efficient-gans)

- ContraGAN: Contrastive Learning for Conditional Image Generation [paper](https://arxiv.org/abs/2006.12681)

- CircleGAN: Generative Adversarial Learning across Spherical Circles  [paper](https://arxiv.org/abs/2011.12486)

- Reconstructing Perceptive Images from Brain Activity by Shape-Semantic GAN  [paper](https://papers.nips.cc/paper/2020/hash/9813b270ed0288e7c0388f0fd4ec68f5-Abstract.html)

- Top-k Training of GANs: Improving GAN Performance by Throwing Away Bad Samples  [paper](https://arxiv.org/abs/2002.06224)

- Training Generative Adversarial Networks with Limited Data  [paper](https://arxiv.org/abs/2006.06676)

- Learning Semantic-aware Normalization for Generative Adversarial Networks  [paper](https://papers.nips.cc/paper/2020/hash/f885a14eaf260d7d9f93c750e1174228-Abstract.html)

- Training Generative Adversarial Networks by Solving Ordinary Differential Equations  [paper](https://arxiv.org/abs/2010.15040)

- Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation  [paper](https://arxiv.org/abs/2010.12136)

- A Decentralized Parallel Algorithm for Training Generative Adversarial Nets [paper](







<h2>ICML 2020</h2>

- Small-GAN: Speeding up GAN Training using Core-Sets
- GANs May Have No Nash Equilibria
- Understanding and Stabilizing GANs' Training Dynamics Using Control Theory
- Unsupervised Discovery of Interpretable Directions in the GAN Latent Space
- SimGANs: Simulator-Based Generative Adversarial Networks for ECG Synthesis to Improve Deep ECG Classification
- Semi-Supervised StyleGAN for Disentanglement Learning
- T-GD: Transferable GAN-generated Images Detection Framework
- Learning disconnected manifolds: a no GAN's land
- InfoGAN-CR: Disentangling Generative Adversarial Networks with Contrastive Regularizers
- NetGAN without GAN: From Random Walks to Low-Rank Approximations
- AutoGAN-Distiller: Searching to Compress Generative Adversarial Networks
- Feature Quantization Improves GAN Training
- SGD Learns One-Layer Networks in WGANs
- Bridging the Gap Between f-GANs and Wasserstein GANs
- Implicit competitive regularization in GANs
- Reliable Fidelity and Diversity Metrics for Generative Models
- Perceptual Generative Autoencoders
- Learning Structured Latent Factors from Dependent Data:A Generative Model Framework from Information-Theoretic Perspective
- Implicit Generative Modeling for Efficient Exploration
- VFlow: More Expressive Generative Flows with Variational Data Augmentation
- A Chance-Constrained Generative Framework for Sequence Optimization
- On Breaking Deep Generative Model-based Defenses and Beyond
- Generative Flows with Matrix Exponential
- Invertible generative models for inverse problems: mitigating representation error and dataset bias
- Generative Adversarial Imitation Learning with Neural Network Parameterization: Global Optimality and Convergence Rate
- Scalable Deep Generative Modeling for Sparse Graphs
- A Generative Model for Molecular Distance Geometry
- Intrinsic Reward Driven Imitation Learning via Generative Model
- On the Power of Compressed Sensing with Generative Models
- Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data
- Sequential Transfer in Reinforcement Learning with a Generative Model
- Optimizing Dynamic Structures with Bayesian Generative Search
- Learning and Simulation in Generative Structured World Models
- Evaluating Lossy Compression Rates of Deep Generative Models
- Learning with Good Feature Representations in Bandits and in RL with a Generative Model
- Fair Generative Modeling via Weak Supervision
- Source Separation with Deep Generative Priors
- Generative Pretraining From Pixels
- Conditional Augmentation for Generative Modeling
- Robust One-Bit Recovery via ReLU Generative Networks: Near-Optimal Statistical Rate and Global Landscape Analysis
- Equivariant Flows: exact likelihood generative learning for symmetric densities.
- PolyGen: An Autoregressive Generative Model of 3D Meshes



<h2>CVPR 2020</h2>

1. Transformation GAN for Unsupervised Image Synthesis and Representation Learning
2. Alleviation of Gradient Exploding in GANs: Fake Can Be Real
3. Learning to Simulate Dynamic Environments With GameGAN
4. Image Processing Using Multi-Code GAN Prior
5. ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation
6. DOA-GAN: Dual-Order Attentive Generative Adversarial Network for Image Copy-Move Forgery Detection and Localization
7. Augmenting Colonoscopy Using Extended and Directional CycleGAN for Lossy Image Translation
8. Cascade EF-GAN: Progressive Facial Expression Editing With Local Focuses
9. GanHand: Predicting Human Grasp Affordances in Multi-Object Scenes
10. Controllable Person Image Synthesis With Attribute-Decomposed GAN
11. PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer
12. GAN Compression: Efficient Architectures for Interactive Conditional GANs
13. The GAN That Warped: Semantic Attribute Editing With Unpaired Data
14. CIAGAN: Conditional Identity Anonymization Generative Adversarial Networks
15. CookGAN: Causality Based Text-to-Image Synthesis
16. MaskGAN: Towards Diverse and Interactive Facial Image Manipulation
17. AdversarialNAS: Adversarial Neural Architecture Search for GANs
18. Regularizing Discriminative Capability of CGANs for Semi-Supervised Generative Learning
19. UCTGAN: Diverse Image Inpainting Based on Unsupervised Cross-Space Translation
20. Editing in Style: Uncovering the Local Semantics of GANs
21. Weakly-Supervised Domain Adaptation via GAN and Mesh Model for Estimating 3D Hand Poses Interacting Objects
22. StyleRig: Rigging StyleGAN for 3D Control Over Portrait Images
23. Copy and Paste GAN: Face Hallucination From Shaded Thumbnails
24. MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks
25. ManiGAN: Text-Guided Image Manipulation
26. Analyzing and Improving the Image Quality of StyleGAN
27. ARShadowGAN: Shadow Generative Adversarial Network for Augmented Reality in Single Light Scenes
28. StarGAN v2: Diverse Image Synthesis for Multiple Domains
29. Image2StyleGAN++: How to Edit the Embedded Images?
30. BachGAN: High-Resolution Image Synthesis From Salient Object Layout
31. On Positive-Unlabeled Classification in GAN
32. Interpreting the Latent Space of GANs for Semantic Face Editing
33. MineGAN: Effective Knowledge Transfer From GANs to Target Domains With Few Images
34. LG-GAN: Label Guided Adversarial Network for Flexible Targeted Attack of Point Cloud Based Deep Networks
35. RiFeGAN: Rich Feature Generation for Text-to-Image Synthesis From Prior Knowledge
36. SurfelGAN: Synthesizing Realistic Sensor Data for Autonomous Driving
37. RL-CycleGAN: Reinforcement Learning Aware Simulation-to-Real
38. Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN
39. StereoGAN: Bridging Synthetic-to-Real Domain Gap by Joint Optimization of Domain Translation and Stereo Matching
40. PuppeteerGAN: Arbitrary Portrait Animation With Semantic-Aware Appearance Transformation
41. Prior Guided GAN Based Semantic Inpainting
42. Synthetic Learning: Learn From Distributed Asynchronized Discriminator GAN Without Sharing Medical Image Data
43. SharinGAN: Combining Synthetic and Real Data for Unsupervised Geometry Estimation
44. PhysGAN: Generating Physical-World-Resilient Adversarial Examples for Autonomous Driving
45. Diverse Image Generation via Self-Conditioned GANs
46. Your Local GAN: Designing Two Dimensional Local Attention Mechanisms for Generative Models



