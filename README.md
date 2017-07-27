# Benchmark for outcome-oriented predictive process monitoring
Supplementary material for the article ["Outcome-Oriented Predictive Process Monitoring: Review and Benchmark"](https://arxiv.org/abs/1707.06766) by Irene Teinemaa, Marlon Dumas, Marcello La Rosa, and Fabrizio Maria Maggi.

This repository provides implementations of different techniques for outcome-oriented predictive business process monitoring. The aim of these techniques is to predict a (pre-defined) binary case outcome of a running (partial) trace. 

The benchmark includes implementations of four sequence encodings (for further description, refer to the paper):

* Static encoding
* Last state encoding
* Aggregation encoding
* Index-based encoding

Moreover, the repository contains implementations of four bucketing methods (see the paper for more details):

* No bucketing
* KNN
* State-based
* Clustering
* Prefix length based

The benchmark experiments have been performed using two classifiers:

* Random forest
* Gradient boosted trees

Together with the code, we make available 10 datasets that were used in the evaluation section in the paper. These datasets correspond to different prediction tasks, formulated on 3 publicly available event logs (namely, the [BPIC 2011](https://data.4tu.nl/repository/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54), [BPIC 2015](http://data.4tu.nl/repository/uuid:31a308ef-c844-48da-948c-305d167a0ec1), and [BPIC 2017](http://data.4tu.nl/repository/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) event logs). These (labeled and preprocessed) benchmark datasets can be found at https://drive.google.com/file/d/0B-ifLE4v5dewMlI5enJKWnBwb0k.

If you use code from this repository, please cite the following paper:

```
@article{Teinemaa2017,
   author = {{Teinemaa}, I. and {Dumas}, M. and {La Rosa}, M. and {Maggi}, F.~M.},
    title = "{Outcome-Oriented Predictive Process Monitoring: Review and Benchmark}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1707.06766},
 primaryClass = "cs.AI",
 keywords = {Computer Science - Artificial Intelligence},
     year = 2017,
    month = jul,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170706766T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
