FootAndBall by Jacek Komorowski, Grzegorz Kurzejamski and Grzegorz Sarwas. Copyright (c) 2020 Sport Algorithmics and Gaming. MIT license retained in LICENSE.FootAndBall.txt.

fpn.py is copied unchanged from https://github.com/jac99/FootAndBall at commit 01261092a93de55026745e532a2e12188ad86801. model.py adapts the official fb1 module hierarchy for safe state-dictionary loading and explicit output tensors. The architecture and all checkpoint parameters are unchanged. No training/data-loader code or pretrained weights are redistributed.

Postprocessing is a new implementation of the official strict eight-neighbour local-maximum rule; the official NMS file credits Kornia. No Kornia implementation is copied into this module.
