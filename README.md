## Model Description

This repository contains the **Teacher and Student implementations** of the proposed cascaded bronchoscopy image classification framework.

The framework consists of two sequential classification stages:

1. **Normal–Abnormal Classification:** Distinguishes normal images from abnormal images.
2. **TB–Cancer Classification:** Further classifies abnormal images as either Tuberculosis (TB) or Cancer.

Both stages use the same Teacher and Student implementations. The only differences between the two classification tasks are the class labels and the corresponding datasets.
Therefore, to avoid redundant code, this repository presents the Teacher and Student implementations using the TB–Cancer classification task as an example. 
The same implementations are also used for the Normal–Abnormal classification task by changing the class labels and dataset.
