#!/bin/env python2.7

import getopt
import os
import sys
import time

from os import path

import cv
import numpy

from scipy.ndimage import convolve
from scipy.signal import medfilt

# kernel for x and y derivation
kx = numpy.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
ky = kx.transpose()

FEATURE_DIM = 9
MEDFILT_KSIZE = 3

def flow_from_video(video_name, flow_name):
    """
    Calculate optical flow feature from a video
    """

    stream = cv.CaptureFromFile(video_name)
    nFrames = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_COUNT))
    nRows = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_HEIGHT))
    nCols = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_WIDTH))

    print 'video: %s' % (video_name)
    print 'number of frame: %d' % (nFrames)

    print 'flow: %s' % (flow_name)
    if path.exists(flow_name):
        print '%s.npy exists' % (flow_name)
    else:
        # for zero padding
        cur_frame_m = cv.CreateMat(nRows, nCols, cv.CV_8UC1)
        cv.Set(cur_frame_m, 0)
        pre_frame_m = cv.CreateMat(nRows, nCols, cv.CV_8UC1)
        cv.Set(pre_frame_m, 0)
        cur_flow_m = cv.CreateMat(nRows, nCols, cv.CV_32FC2)
        cv.Set(cur_flow_m, 0)
        pre_flow_m = cv.CreateMat(nRows, nCols, cv.CV_32FC2)
        cv.Set(pre_flow_m, 0)

        # where all the flow features are saved
        all_feature = numpy.empty((FEATURE_DIM, 0), 'float32')

        sys.stdout.write('%04d' % (0))
        for i in range(nFrames):
            sys.stdout.write('\b\b\b\b%04d' % (i))
            sys.stdout.flush()

            # calculate the optical flow
            pre_frame_m, cur_frame_m = cur_frame_m, pre_frame_m
            frame = cv.QueryFrame(stream)
            #TODO: add exception safe code here
            cv.CvtColor(frame, cur_frame_m, cv.CV_BGR2GRAY)

            pre_flow_m, cur_flow_m = cur_flow_m, pre_flow_m
            cv.CalcOpticalFlowFarneback(
                pre_frame_m, cur_frame_m, cur_flow_m, pyr_scale=0.5, levels=3,
                winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            # convert OpenCV Mat to Np array
            cur_frame = numpy.asarray(cur_frame_m, 'float32')
            pre_frame = numpy.asarray(pre_frame_m, 'float32')
            cur_flow = numpy.asarray(cur_flow_m, 'float32')
            pre_flow = numpy.asarray(pre_flow_m, 'float32')

            # split flow vector to u and v
            cur_u = cur_flow[:, :, 0]
            cur_v = cur_flow[:, :, 1]
            pre_u = pre_flow[:, :, 0]
            pre_v = pre_flow[:, :, 1]

            # median filter the optical flow
            #TODO: should the median filter done separately for u and v or not?
            #TODO: median filter for pre_u and pre_v are not necessary
            #      if pre_u, pre_v, cur_u and cur_v are defined outside for loop
            cur_u = medfilt(cur_u, MEDFILT_KSIZE)
            cur_v = medfilt(cur_v, MEDFILT_KSIZE)
            pre_u = medfilt(pre_u, MEDFILT_KSIZE)
            pre_v = medfilt(pre_v, MEDFILT_KSIZE)

            # temporal derivative
            It = pre_frame - cur_frame
            ut = pre_u - cur_u
            vt = pre_v - cur_v

            # spatial derivative
            # 1st order x|y of derivative of u|v
            dux = convolve(cur_u, kx)
            duy = convolve(cur_u, ky)
            dvx = convolve(cur_v, kx)
            dvy = convolve(cur_v, ky)

            # Div, Vor
            div = dux + dvy
            vor = dvx - duy

            # Gten, Sten
            g_ten = dux * dvy - duy * dvx
            s_ten = dux * dvy - pow((duy + dvx) / 2, 2)

            # Join features from current frame together
            cur_feature = numpy.concatenate(
                (cur_u.reshape((1, nRows, nCols)),
                 cur_v.reshape((1, nRows, nCols)),
                 It.reshape((1, nRows, nCols)),
                 ut.reshape((1, nRows, nCols)),
                 vt.reshape((1, nRows, nCols)),
                 div.reshape((1, nRows, nCols)),
                 vor.reshape((1, nRows, nCols)),
                 g_ten.reshape((1, nRows, nCols)),
                 s_ten.reshape((1, nRows, nCols))))

            # reshpae the feature from current frame
            cur_feature = cur_feature.reshape(9, -1);

            # Append it to the all_feature pool
            all_feature = numpy.hstack((all_feature, cur_feature))

        # calculate the final feature: the covariance matrix (9 x 9)
        cov_feature = numpy.cov(all_feature)
        # take only the upper triangle
        feature = numpy.triu(cov_feature)
        feature = feature[feature.nonzero()]

        # save feature to file
        numpy.save(flow_name, feature)
        print '\nsave result: %s' % (flow_name)


def process(type, input_dir, output_dir):
    log_name = './log' + time.strftime('%m%d%H%M') + type
    log = open(log_name, 'w')
    log_write = 0

    if type == 'K':
        # KTH dataset

        if input_dir is '':
            input_dir = '/home/zhanwu/storage/KTH/video'
        if output_dir is '':
            output_dir = '/home/zhanwu/storage/KTH/flow'

        action_names = ['boxing', 'handclapping', 'handwaving', 'jogging',
                        'running', 'walking']

        # create, if output dir doesn't exist
        if not path.exists(output_dir):
            print 'create dir %s' % (output_dir)
            os.mkdir(output_dir)
        # create, if output sub dir doesn't exist
        for action in action_names:
            flow_dir = path.join(output_dir, action)
            if not path.exists(flow_dir):
                print 'create dir %s' % (flow_dir)
                os.mkdir(flow_dir)

        for action in action_names:
            video_dir = path.join(input_dir, action)
            flow_dir = path.join(output_dir, action)

            video_names = os.listdir(video_dir)
            for video_name in video_names:
                video_path = path.join(video_dir, video_name)
                flow_path = path.join(flow_dir, video_name)[:-4] + '.npy'

                try:
                    flow_from_video(video_path, flow_path)
                except:
                    log.writelines(video_path)
                    if log_write is not 1:
                        log_write = 1

    log.close();
    if log_write is not 1:
        os.remove(log_name)


def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hdt:', ['help','debug'])
    except getopt.error, msg:
        print msg
        sys.exit(2)

    for o, a in opts:
        if o in ('-h', '--help'):
            print __doc__
            sys.exit(0)

    for o, a in opts:
        if o in ('-d', '--debug'):
            flow_from_video(('/home/zhanwu/storage/KTH/video/running/person02'
                             '_running_d2_uncomp.avi'), './test')
            sys.exit(0)

    input_dir = ''
    output_dir = ''
    for o, a in opts:
        if o == '-i':
            input_dir = a
        elif o in '-o':
            output_dir = a

    for o, a in opts:
        if o == '-t':
            process(a, input_dir, output_dir)

if __name__ == "__main__":
    main()
