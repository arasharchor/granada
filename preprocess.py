#!/usr/bin/env python2.7

import getopt
import os
import sys
import time
from collections import deque
from os import path

import cv
import numpy as np
from numpy import linalg
from scipy import ndimage

from pdb import set_trace

# Fist note:
#   x, u, row, h, height are for first (vertical) dimension
#   y, v, col, w, width are for second (horinzontal) dimension

# Kernel for x and y derivation
KX = np.array([[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]])
KY = KX.transpose()

# Optical flow parameters explaination from OpenCV:
#   pyrScale: 0.5
#     Classical pyramid, where next layer is twice smaller than the previous.
#   levels: 3
#     The number of pyramid layers, including the initial image.
#   winsize: 15
#     The averaging window size. Larger values increase the algorithm robustness
#     but yield more blurred motion field
#   iterations: 3
#     The number of iterations the algorithm does at each pyramid level
#   polyN: 5
#     Size of the pixel neighborhood used to find polynomial expansion at each
#     pixel. Typically, polyN=5 or 7
#   polySigma: 1.1
#     Standard deviation of the Gaussian that is used to smooth derivatives that
#     are used as a basis for the polynomial expansion. For polyN=5 you can set
#     polySigma=1.1 , for polyN=7 a good value would be polySigma=1.5
#   flags:
#     OPTFLOW_USE_INITIAL_FLOW      4
#     OPTFLOW_FARNEBACK_GAUSSIAN    256
OF_SC = 0.5
OF_LV = 3
OF_WS = 15
OF_IT = 3
OF_PN = 5
OF_SG = 1.1
#TODO: which flag to use?
OF_FG = 0

# Feature dimension: 12
#   [x, y, t, It, u, v, ut, vt, Div, Vor, Gten, Sten]
DIM_FEATURE = 12

# Threshold for selecting moving object:
#   temporal derivation is bigger than threshold
It_theta = 30


def draw_flow(flow, frame, step=10):
    """ Returns a nice representation of optical flow """

    color = (0, 255, 0)
    frame_flow = cv.CreateMat(frame.height, frame.width, cv.CV_8UC3)
    cv.CvtColor(frame, frame_flow, cv.CV_GRAY2BGR)

    for y in xrange(step, flow.height, step):
        for x in xrange(step, flow.width, step):
            fx = int(flow[y, x][0] * 2)   # to make to flow more obvious
            fy = int(flow[y, x][1] * 2)
            cv.Line(frame_flow, (x,y), (x+fx,y+fy), color)
            cv.Circle(frame_flow, (x,y), 2, color, -1)
    return frame_flow


def process_video(video_name, output_prefix='', seg_length=20, overlap=4):
    """
    Calculate features based on optical flow for each segment of the video

    video_name         : full path of the video
    output_prefix      : name prefix and full path of where to save the feature
    seg_length         : length of a segment, default 20
    overlap            : overlap between consecutive segments, default 4

    Videos are divided into several temporal segmentation, with fixed length
    (seg_length) and overlapping (overlap). Features are calculated for each
    segment, thereby one video will generate a sequence of output file.
    """

    video = cv.CaptureFromFile(video_name)
    nFrames = int(cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FRAME_COUNT))
    nRows = int(cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FRAME_HEIGHT))
    nCols = int(cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FRAME_WIDTH))

    # Dump out video information
    print 'video: %s' % (video_name)
    print 'number of frames: %d' % (nFrames)
    print 'video size: %d x %d' %(nRows, nCols)

    # Output files are named by the name prefix followed by 3-digit indices
    output_name = '%s%03d.npy' % (output_prefix, 0)

    # Skip finished video
    if path.exists(output_name):
        print '%s has already finished, skip' % (video_name)
    else:
        # Here goes the main calculating:
        #   For each frame:
        #     1, Calculate the optical flow
        #     2, Calculate different moments (features) of optical flow
        #
        #   For every %seg_length frames (one segment):
        #     1, Join all the features from each frame
        #     2, Calculate the covariance matrix of the all the features
        #     3, Calculate the logarithm matrix of the covariance matrix
        #     4, Save result from step 3

        # For optical flow and the temporal derivations of optical flow and
        # video require information from previous frame, do zero padding
        # for the first frame; OpenCV data format also requires memory
        # allocation in advance
        cur_frame_m = cv.CreateMat(nRows, nCols, cv.CV_8UC1)   #_m for cv.Mat
        cv.Set(cur_frame_m, 0)
        cur_frame = np.asarray(cur_frame_m, 'float32')

        pre_frame_m = cv.CreateMat(nRows, nCols, cv.CV_8UC1)   # ..
        cv.Set(pre_frame_m, 0)
        pre_frame = np.asarray(pre_frame_m, 'float32')

        cur_u = np.zeros((nRows, nCols), 'float32')
        cur_v = np.zeros((nRows, nCols), 'float32')
        pre_u = np.zeros((nRows, nCols), 'float32')
        pre_v = np.zeros((nRows, nCols), 'float32')

        # Feature shared by every frame
        x = np.tile(np.arange(nRows), [nCols, 1])
        x = x.transpose()
        x = x.reshape((nRows, nCols, 1))
        y = np.tile(np.arange(nCols), [nRows, 1])
        y = y.reshape((nRows, nCols, 1))
        t = np.ones([nRows, nCols])
        t = t.reshape((nRows, nCols, 1))     # multiply frame index when use

        # Container for storing all features of a segment
        seg_feature = np.empty((DIM_FEATURE, 0), 'float32')
        iSeg = 0

        # Number of features in last 4 frames
        nFeature_l4_frames = deque()

        for i in range(nFrames):
            # get image frame
            pre_frame_m, cur_frame_m = cur_frame_m, pre_frame_m
            frame = cv.QueryFrame(video)
            cv.CvtColor(frame, cur_frame_m, cv.CV_BGR2GRAY)

            # get optical flow
            cur_flow_m = cv.CreateMat(nRows, nCols, cv.CV_32FC2)
            cv.CalcOpticalFlowFarneback(pre_frame_m, cur_frame_m, cur_flow_m,
                                        OF_SC, OF_LV, OF_WS, OF_IT, OF_PN,
                                        OF_SG, OF_FG)

            # Temporal Derivation of frame
            cur_frame, pre_frame = pre_frame, cur_frame
            cur_frame = np.asarray(cur_frame_m, 'float32')
            It = pre_frame - cur_frame

            # Flow vector
            cur_flow = np.asarray(cur_flow_m, 'float32')
            cur_u, pre_u = pre_u, cur_u
            cur_u = cur_flow[:, :, 0]
            cur_v, pre_v = pre_v, cur_v
            cur_v = cur_flow[:, :, 1]

            # Temporal derivation of flow vector
            ut = pre_u - cur_u
            vt = pre_v - cur_v

            # Spatial derivation of flow vector
            # 1st order x|y of derivative of u|v
            dux = ndimage.convolve(cur_u, KX)
            duy = ndimage.convolve(cur_u, KY)
            dvx = ndimage.convolve(cur_v, KX)
            dvy = ndimage.convolve(cur_v, KY)

            # Div, Vor
            div = dux + dvy
            vor = dvx - duy

            # Gten, Sten
            g_ten = dux * dvy - duy * dvx
            s_ten = dux * dvy - pow((duy + dvx), 2) / 4

            # Join features from current frame together
            # frame_feature size: [nRows, nCols, 12]
            frame_feature = np.concatenate(
                (x,
                 y,
                 t * i,
                 cur_u.reshape((nRows, nCols, 1)),
                 cur_v.reshape((nRows, nCols, 1)),
                 It.reshape((nRows, nCols, 1)),
                 ut.reshape((nRows, nCols, 1)),
                 vt.reshape((nRows, nCols, 1)),
                 div.reshape((nRows, nCols, 1)),
                 vor.reshape((nRows, nCols, 1)),
                 g_ten.reshape((nRows, nCols, 1)),
                 s_ten.reshape((nRows, nCols, 1))),
                axis=2)

            # Select feature from frame which belongs to moving object
            frame_feature = frame_feature[np.abs(It) > It_theta]
            frame_feature = frame_feature.transpose()

            # Append frame feature to the segment feature pool
            seg_feature = np.hstack((seg_feature, frame_feature))

            # Update the nFeature_l4_frames
            if len(nFeature_l4_frames) == 4:
                nFeature_l4_frames.popleft()
                nFeature_l4_frames.append(frame_feature.shape[1])

            # Check if a segment finishs
            if ((i - seg_length) % (seg_length - overlap)) is 0 and i > overlap:

                # 2, Calculate the covariance matrix of the all the features
                cov_feature = np.cov(seg_feature)

                # 3, Calculate the logarithm matrix of the covariance matrix
                cov_u, cov_s, cov_v = linalg.svd(cov_feature);
                cov_s_log = np.log(cov_s)
                log_feature = np.dot(np.dot(cov_u, np.diag(cov_s_log)),
                                        cov_v)

                # Take only the upper triangle (the matrix is symmetry)
                triu_idx = np.ones(cov_feature.shape)
                triu_idx = np.triu(triu_idx)
                log_feature = log_feature[triu_idx.nonzero()]

                # 4, Save result from step 3
                output_name = '%s%03d.npy' % (output_prefix, iSeg)
                np.save(output_name, log_feature)
                print 'Save result: %s' % (output_name)

                # Construct the new segment
                iSeg += 1
                seg_feature = seg_feature[:, -sum(nFeature_l4_frames):]

        print 'Video: %s done' % (video_name)


# # For debugging purpose
# if __name__ == "__main__":
#     process_video('/home/zhanwu/storage/KTH/video2/person13_walking_d1_uncomp.avi')


def main():
    # Log file to record error
    log_name = './log' + time.strftime('%m%d%H%M')
    log = open(log_name, 'w')
    log_write = 0

    # Parse command line options
    opts, args = getopt.getopt(sys.argv[1:], 'io:')

    input_dir = ''
    output_dir = ''
    for o, a in opts:
        if o == '-i':
            input_dir = a
        elif o in '-o':
            output_dir = a

    if input_dir is '':
        input_dir = '/home/zhanwu/storage/KTH/video'
    if output_dir is '':
        output_dir = '/home/zhanwu/storage/KTH/feature_' + time.strftime('%m%d')

    action_names = ['boxing', 'handclapping', 'handwaving', 'jogging',
                    'running', 'walking']

    # Create, if output dir doesn't exist
    if not path.exists(output_dir):
        print 'Create dir %s' % (output_dir)
        os.mkdir(output_dir)
    # Create, if output sub dir doesn't exist
    for action in action_names:
        flow_dir = path.join(output_dir, action)
        if not path.exists(flow_dir):
            print 'Create dir %s' % (flow_dir)
            os.mkdir(flow_dir)

    for action in action_names:
        video_dir = path.join(input_dir, action)
        feature_dir = path.join(output_dir, action)

        video_names = os.listdir(video_dir)
        for video_name in video_names:
            video_path = path.join(video_dir, video_name)
            feature_path_pre = path.join(feature_dir, video_name)[:-4]

            # process_video(video_path, feature_path_pre)
            try:
                process_video(video_path, feature_path_pre)
            except:
                #TODO: clean up the feature files for unfinished video
                log.writelines(video_path)
                log_write = 1

    log.close();
    if log_write is not 1:
        os.remove(log_name)


if __name__ == "__main__":
    main()
