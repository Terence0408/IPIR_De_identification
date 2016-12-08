#!/usr/bin/python
# -*- coding: utf-8 -*-

# Goal: Build label prediction layer and label sequence optimization layer to predict PHI type.
# Step: 1. Build bidirectional lstm layer.
#             Input: token(100 dimension) sequence of a sentence.
#             Output: 100 dimension.
#       2. Add dense layer
#             Input: 100 dimension(lstm layer result)
#             Output: 19 prob. dimension
#       3. Add label sequence optimization layer
#             Input: 19 prob. dimension(dense layer result)
#             Output: 19 PHI label(binary vector)
#       Train step: x: token(100 dimension) sequence of a sentence.
#                   y: 19 PHI label(18 PHI-type and 1 non PHI)