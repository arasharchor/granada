#!/usr/bin/env python2.7

import getopt
import os
import sys
import time
from os import path

import cv
import numpy
from collections import deque
from numpy import linalg
from scipy import ndimage

from pdb import set_trace

# Fist note:
#   x, u, row, h, height are for first (vertical) dimension
#   y, v, col, w, width are for second (horinzontal) dimension

# kernel for x and y derivation
KX = numpy.array([[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]])
KY = KX.transpose()

# optical flow parameters explaination from OpenCV:
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

# feature dimension: 12
#   [x, y, t, It, u, v, ut, vt, Div, Vor, Gten, Sten]
DIM_FEATURE = 12


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


def process_video(video_name, feature_name_prefix='', seg_length=20, overlap=4):
    """
    Calculate features based on optical flow for each segment of the video

    video_name         : full path of the video
    feature_name_prefix: name prefix and full path of where to save the feature
    seg_length         : length of a segment, default 20
    overlap            : overlap between consecutive segments, default 4
    """

    video = cv.CaptureFromFile(video_name)
    nFrames = int(cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FRAME_COUNT))
    nRows = int(cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FRAME_HEIGHT))
    nCols = int(cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FRAME_WIDTH))

    # dump out video information
    print 'video: %s' % (video_name)
    print 'number of frames: %d' % (nFrames)

    # features files are named by the name prefix followed by a 3-digit number
    feature_name = '%s%03d.npy' % (feature_name_prefix, 0)

    # for the process may be interupt,
    # when resume, it doesn't need to redo finished videos
    if path.exists(feature_name):
        print '%s exists, skip' % (feature_name)
    else:
        # Here goes the main calculating:
        #   For each frame, first calculate the optical flow and then get the
        #   features based on it.
        #
        #   Note that many features requires data from previous frame as well

        # zero padding
        cur_frame_m = cv.CreateMat(nRows, nCols, cv.CV_8UC1)
        cv.Set(cur_frame_m, 0)
        cur_frame = numpy.asarray(cur_frame_m, 'float32')

        pre_frame_m = cv.CreateMat(nRows, nCols, cv.CV_8UC1)
        cv.Set(pre_frame_m, 0)
        pre_frame = numpy.asarray(pre_frame_m, 'float32')

        cur_u = numpy.zeros((nRows, nCols), 'float32')
        cur_v = numpy.zeros((nRows, nCols), 'float32')
        pre_u = numpy.zeros((nRows, nCols), 'float32')
        pre_v = numpy.zeros((nRows, nCols), 'float32')

        # index of current segmentation
        iSeg = 0

        # shared feature by every frame
        x = numpy.tile(numpy.arange(nRows), [nCols, 1])
        x = x.transpose()
        x = x.reshape((nRows, nCols, 1))
        y = numpy.tile(numpy.arange(nCols), [nRows, 1])
        y = y.reshape((nRows, nCols, 1))
        t = numpy.ones([nRows, nCols])
        t = t.reshape((nRows, nCols, 1))

        sys.stdout.write('%04d' % (0))
        for i in range(nFrames):
            sys.stdout.write('\b\b\b\b%04d' % (i))
            sys.stdout.flush()

            # get image frame
            pre_frame_m, cur_frame_m = cur_frame_m, pre_frame_m
            frame = cv.QueryFrame(video)
            cv.CvtColor(frame, cur_frame_m, cv.CV_BGR2GRAY)

            # get optical flow
            cur_flow_m = cv.CreateMat(nRows, nCols, cv.CV_32FC2)
            cv.CalcOpticalFlowFarneback(pre_frame_m, cur_frame_m, cur_flow_m,
                                        OF_SC, OF_LV, OF_WS, OF_IT, OF_PN,
                                        OF_SG, OF_FG)

            # get frame in numpy format
            cur_frame, pre_frame = pre_frame, cur_frame
            cur_frame = numpy.asarray(cur_frame_m, 'float32')

            # get flow vector in numpy format
            cur_flow = numpy.asarray(cur_flow_m, 'float32')
            cur_u, pre_u = pre_u, cur_u
            cur_u = cur_flow[:, :, 0]
            cur_v, pre_v = pre_v, cur_v
            cur_v = cur_flow[:, :, 1]

            # temporal derivative
            It = pre_frame - cur_frame
            ut = pre_u - cur_u
            vt = pre_v - cur_v

            # spatial derivative
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
            cur_feature = numpy.concatenate(
                (x, y, t * i,
                 cur_u.reshape((nRows, nCols, 1)),
                 cur_v.reshape((nRows, nCols, 1)),
                 It.reshape((nRows, nCols, 1)),
                 ut.reshape((nRows, nCols, 1)),
                 vt.reshape((nRows, nCols, 1)),
                 div.reshape((nRows, nCols, 1)),
                 vor.reshape((nRows, nCols, 1)),
                 g_ten.reshape((nRows, nCols, 1)),
                 s_ten.reshape((nRows, nCols, 1))), axis=2)

            thd = 30
            cur_feature_m = cur_feature[numpy.abs(It) > thd]

            cur_feature = cur_feature.reshape(-1, DIM_FEATURE);
            cur_feature = cur_feature.transpose()
            cur_feature_m = cur_feature_m.transpose()

            # # for debugging only
            # cv.ShowImage('video', frame)
            # cv.ShowImage('It', cv.fromarray(numpy.abs(It)))
            # cv.ShowImage('flow', draw_flow(cur_flow_m, cur_frame_m))
            # cv.ShowImage('u', cv.fromarray(numpy.abs(cur_u)))
            # cv.ShowImage('v', cv.fromarray(numpy.abs(cur_v)))
            # cv.ShowImage('ut', cv.fromarray((cur_u - cur_v)))
            # cv.ShowImage('vt', cv.fromarray(numpy.abs(vt)))
            # cv.ShowImage('dux', cv.fromarray(numpy.abs(dux)))
            # cv.ShowImage('duy', cv.fromarray(numpy.abs(duy)))
            # cv.ShowImage('dvx', cv.fromarray((numpy.abs(g_ten))))
            # cv.ShowImage('dvy', cv.fromarray((numpy.abs(s_ten))))

            # cv.MoveWindow('video', 0, 0)
            # cv.MoveWindow('video2', 350, 0)
            # cv.MoveWindow('It', 700, 0)
            # cv.MoveWindow('flow', 1050, 0)
            # cv.MoveWindow('u', 0, 350)
            # cv.MoveWindow('v', 350, 350)
            # cv.MoveWindow('ut', 700, 350)
            # cv.MoveWindow('vt', 1050, 350)
            # cv.MoveWindow('dux', 0, 700)
            # cv.MoveWindow('duy', 350, 700)
            # cv.MoveWindow('dvx', 700, 700)
            # cv.MoveWindow('dvy', 1050, 700)

            # cv.WaitKey(30)

            if i is 0:
                # very beginning, construct the all_feature pool
                all_feature = numpy.empty((DIM_FEATURE, 0), 'float32')
                all_feature_m = numpy.empty((DIM_FEATURE, 0), 'float32')

                nFeature_last_4_frames = deque()

            if ((i - seg_length) % (seg_length - overlap)) is 0 and i > overlap:
                # beginning of a new temporal segment
                # 1, compute the covariance of the last segment
                # 2, construct the new segment

                # 1, compute the logarithm covariance of the last segment
                cov_feature = numpy.cov(all_feature)
                cov_u, cov_s, cov_v = linalg.svd(cov_feature);
                cov_s = numpy.log(cov_s)
                cov_feature = cov_u * numpy.diag(cov_s) * cov_v

                # take only the upper triangle
                idx = numpy.ones(cov_feature.shape)
                idx = numpy.triu(idx)
                feature = cov_feature[idx.nonzero()]

                # save covariance to file
                feature_name = '%s_f_%03d.npy' % (feature_name_prefix, iSeg)
                numpy.save(feature_name, feature)
                print '\nsave result: %s' % (feature_name)

                # 2, construct the new segment
                all_feature = all_feature[:, -4 * 120 * 160:]

                # same task for all_feature_m
                if all_feature_m.size > DIM_FEATURE:
                    # 1, compute the logarithm covariance of the last segment
                    cov_feature_m = numpy.cov(all_feature_m)
                    cov_u_m, cov_s_m, cov_v_m = linalg.svd(cov_feature_m);
                    cov_s_m = numpy.log(cov_s_m)
                    cov_feature_m = cov_u_m * numpy.diag(cov_s_m) * cov_v_m

                    # take only the upper triangle
                    idx = numpy.ones(cov_feature_m.shape)
                    idx = numpy.triu(idx)
                    feature_m = cov_feature_m[idx.nonzero()]

                    # save covariance to file
                    feature_m_name = '%s_m_%03d.npy' % (feature_name_prefix, iSeg)
                    numpy.save(feature_m_name, feature_m)
                    print '\nsave result: %s' % (feature_m_name)

                    # 2, construct the new segment
                    sumFeature_last_4_frames = 0
                    for nFeature_one_frame in nFeature_last_4_frames:
                        sumFeature_last_4_frames += sumFeature_last_4_frames

                    all_feature_m = all_feature_m[:, -sumFeature_last_4_frames:]

                iSeg += 1

            # append current feature to the all_feature pool
            all_feature = numpy.hstack((all_feature, cur_feature))
            all_feature_m = numpy.hstack((all_feature, cur_feature_m))

            # to remember how many features in the last 4 frames
            # it is used when we construct the new feature pool
            if len(nFeature_last_4_frames) == 4:
                nFeature_last_4_frames.popleft()
            nFeature_last_4_frames.append(cur_feature_m.shape[1])

            #TODO: the last temporal segment is discarded
            #      need it to be processed as well?

        sys.stdout.write('\n')
        print 'video: %s done' % (video_name)


# if __name__ == "__main__":
#     process_video('/home/zhanwu/storage/KTH/video2/person13_walking_d1_uncomp.avi')


def main():
    # log file to record error
    log_name = './log' + time.strftime('%m%d%H%M')
    log = open(log_name, 'w')
    log_write = 0

    # parse command line options
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
        output_dir = '/home/zhanwu/storage/KTH/flow_0821'

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
            flow_path_pre = path.join(flow_dir, video_name)[:-4]

            try:
                process_video(video_path, flow_path_pre, 20, 4)
            except:
                #TODO: clean up the feature files for unfinished video
                log.writelines(video_path)
                log_write = 1

    log.close();
    if log_write is not 1:
        os.remove(log_name)


if __name__ == "__main__":
    main()
