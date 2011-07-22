#!/usr/bin/env python2.7

import os
from os import path

import numpy
from matplotlib import mlab
from matplotlib import pyplot
import mpl_toolkits.mplot3d.axes3d as p3

if __name__ == "__main__":
    output_dir = '/home/zhanwu/storage/KTH/flow'

    action_names = ['boxing', 'handclapping', 'handwaving', 'jogging',
                    'running', 'walking']

    all_f = numpy.empty((0, 45), 'float32')
    for action in action_names:
        flow_dir = path.join(output_dir, action)

        flow_names = os.listdir(flow_dir)
        i = 0;
        for flow_name in flow_names:
            if i < 500:
                flow_path = path.join(flow_dir, flow_name)
                f = numpy.load(flow_path)
                all_f = numpy.vstack((all_f, f.reshape(-1, 45)))
                i += 1
            else:
                break

    all_f = all_f.transpose()
    [pcomponent, trans, fracVar] = mlab.prepca(all_f)

    # take first 3 component
    pca = pcomponent[0:3, :]


    fig = pyplot.figure()
    c = ['red', 'blue', 'purple', 'green', 'black', 'pink']
    ax = p3.Axes3D(fig)
    for i in xrange(6):
        ax.scatter(pca[0, i*500:(i+1)*500], pca[1, i*500:(i+1)*500], pca[2, i*500:(i+1)*500], marker='x', color=c[i])
    pyplot.show()
