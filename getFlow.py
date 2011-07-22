#!/usr/bin/env python2.7

import getopt
import os
import sys
import time

from os import path

import cv
import numpy

from scipy import ndimage      # convolve


# kernel for x and y derivation
KX = numpy.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
KY = KX.transpose()
DIM_FEATURE = 9


def flow_from_video(video_name, flow_name_pre, seg_length=20, overlap=4):
    """
    Calculate optical flow feature from a video
    """

    stream = cv.CaptureFromFile(video_name)
    nFrames = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_COUNT))
    nRows = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_HEIGHT))
    nCols = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_WIDTH))

    print 'video: %s' % (video_name)
    print 'number of frame: %d' % (nFrames)

    flow_name = '%s%03d.npy' % (flow_name_pre, 0)
    if path.exists(flow_name):
        print '%s exists, skip' % (flow_name)
    else:
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

        iSeg = 0

        sys.stdout.write('%04d' % (0))
        for i in range(nFrames):
            sys.stdout.write('\b\b\b\b%04d' % (i))
            sys.stdout.flush()

            # get image frame
            pre_frame_m, cur_frame_m = cur_frame_m, pre_frame_m
            frame = cv.QueryFrame(stream)
            cv.CvtColor(frame, cur_frame_m, cv.CV_BGR2GRAY)

            # get optical flow
            #TODO: avoid using magic numbers
            cur_flow_m = cv.CreateMat(nRows, nCols, cv.CV_32FC2)
            cv.CalcOpticalFlowFarneback(
                pre_frame_m, cur_frame_m, cur_flow_m, pyr_scale=0.5, levels=3,
                winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

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
            cur_feature = cur_feature.reshape(DIM_FEATURE, -1);

            if i is 0:
                # very beginning, construct the all_feature pool
                all_feature = numpy.empty((DIM_FEATURE, 0), 'float32')

            if ((i - seg_length) % (seg_length - overlap)) is 0 and i > overlap:
                # beginning of a new temporal segment
                # 1, compute the covariance of the last segment
                # 2, construct the new segment

                # 1, compute the covariance of the last segment
                cov_feature = numpy.cov(all_feature)

                # take only the upper triangle
                feature = numpy.triu(cov_feature)
                feature = feature[feature.nonzero()]

                # save covariance to file
                flow_name = '%s%03d.npy' % (flow_name_pre, iSeg)
                numpy.save(flow_name, feature)
                print '\nsave result: %s' % (flow_name)

                # 2, construct the new segment
                all_feature = all_feature[:, -4:]

                iSeg += 1

            # append current feature to the all_feature pool
            all_feature = numpy.hstack((all_feature, cur_feature))

            #TODO: the last temporal segment is discarded
            #      need it to be processed as well?

        sys.stdout.write('\n')
        print 'video: %s done' % (video_name)


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
            flow_path_pre = path.join(flow_dir, video_name)[:-4]

            try:
                flow_from_video(video_path, flow_path_pre, 20, 4)
            except:
                log.writelines(video_path)
                log_write = 1

    log.close();
    if log_write is not 1:
        os.remove(log_name)


if __name__ == "__main__":
    main()
