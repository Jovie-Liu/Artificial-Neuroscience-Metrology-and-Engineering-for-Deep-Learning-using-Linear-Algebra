# Artificial-Neuroscience-Metrology-and-Engineering-for-Deep-Learning-using-Linear-Algebra

- The project titled *Artificial Neuroscience: Metrology and Engineering for Deep Learning Using Linear Algebra* is a UKRI (UK Research and Innovation) funded EPSRC (Engineering and Physical Sciences Research Council) Discipline Hopping in ICT Grant featuring collaborations between the School of Electronic Engineering and Computer Science and the School of Mathematical Sciences at Queen Mary University of London, United Kingdom. Associated with the Centre for Digital Music ([C4DM](https://www.c4dm.eecs.qmul.ac.uk/people/)) and the [Centre for Fundamentals of AI and Computational Theory](https://www.seresearch.qmul.ac.uk/cfcs/people/jiliu/), I work as a Postdoctoral Research Associate co-supervised by [Prof. Mark Sandler](https://www.seresearch.qmul.ac.uk/cmai/people/msandler/#grants) and [Prof. Boris Khoruzhenko](https://www.seresearch.qmul.ac.uk/cpsd/people/bkhoruzhenko/#grants).

- To gain a deeper understanding of the **correlation structure in audio processing neural networks**, we apply Empirical Spectral Density (ESD) techniques to weight matrices to investigate the dynamics of training.

- To build more **efficient, compact, and less energy-consuming neural networks**, we employ tensor decomposition on the weight matrices and examine the effects of low-rank approximation on the neural network behaviors.

---

## Research Phase I: Low-Rank Structure in Linear Weights

In the first phase of this project, we investigate the low-rank structure of the linear weights in [**Conv-TasNet**](https://arxiv.org/pdf/1809.07454), focusing on the pointwise convolution layers ([Video Demo](https://www.youtube.com/watch?v=fL-FDF-Iojk&list=PLWSd-mlbNCAWjovFmisi1asUd0StPzdPc&index=2)). Specifically, we study whether singular value decomposition (SVD) can reveal redundant directions in these learned weights, allowing us to compress the model while preserving its source separation performance.

### [Experimental Procedure](https://github.com/Jovie-Liu/Artificial-Neuroscience-Metrology-and-Engineering-for-Deep-Learning-using-Linear-Algebra/blob/main/Document%201_1%20Low%20Rank%20Structure%20on%20Linear%20Weights%20%5BFull%5D.ipynb)

1. **Estimate the effective rank of each linear weight**  
   For every pointwise convolution weight matrix, we compute its SVD, estimate its effective rank, and truncate the matrix accordingly.

2. **Replace the original full weights with truncated weights**  
   We substitute the original dense weights with their truncated low-rank approximations and evaluate the model immediately.

   - **Original Model Validation Loss:** -14.21  
   - **Truncated Model Validation Loss:** -14.00  
   - **Original Model Training Loss:** -16.24  
   - **Truncated Model Training Loss:** -15.92  

   The performance degrades only slightly after truncation. This indicates that the original layers contain **substantial redundancy**, and that the model can be made significantly **smaller and more compact without sacrificing much effectiveness**.

3. **Fine-tune the low-rank model**  
   After fine-tuning the truncated model, we get
   
   - **Fine-Tuned Truncated Model Validation Loss:** -14.23
   
   Slightly better than the original full model (Validation Loss: -14.21).

This result suggests that low-rank reduction is not merely a compression technique for reducing parameter count. More importantly, it may uncover a **better structural form** of the weight matrix—one that removes redundant directions, preserves the most informative components, and yields a more efficient representation for the task. In this sense, low-rank structure may serve not only as a tool for model compression, but also as a form of **inductive bias** that improves optimization and generalization.
