Benchmarks Overview
===================

The table summarizes key properties of each supported benchmark.

=================  ===============================  ================================  ==================
Benchmark           Datasets                         Primary metrics                   Training epochs
=================  ===============================  ================================  ==================
NASBench-101       CIFAR-10                         train/val/test accuracy           4, 12, 36, 108
NASBench-201       CIFAR-10, CIFAR-100,             train/val/test accuracy, losses   12, 200
                    ImageNet16-120                                                       
NASBench-301       CIFAR-10, CIFAR-100              surrogate val/test accuracy       N/A (surrogate)
=================  ===============================  ================================  ==================

Notes:
- NASBench-301 uses surrogate models; epochs are not applicable.

