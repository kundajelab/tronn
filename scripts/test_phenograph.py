#!/usr/bin/env python3

import phenograph
import numpy as np


data = np.random.rand(100, 100)

communities, graph, Q = phenograph.cluster(data)
