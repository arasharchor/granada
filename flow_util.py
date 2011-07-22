#!/usr/bin/env python2.7

import sys

import cv
import numpy

from scipy.signal import medfilt

def draw_flow(window, image , flow):
    INTERVAL = 2
    COLOR_0 = cv.RGB(255, 0, 0)
    COLOR_1 = cv.RGB(0, 255, 0)
    COLOR_2 = cv.RGB(0, 0, 255)
    COLOR_3 = cv.RGB(0, 0, 0)
    COLOR = (COLOR_0, COLOR_1, COLOR_2, COLOR_3)

    for y in xrange(0, flow.height):
        for x in xrange(0, flow.width):
            fx, fy = flow[y, x]
            fx = int(fx)
            fy = int(fy)
            if fx is not 0 or fy is not 0:
                color = COLOR[((fx > 0) << 1) + (fy > 0)]
                cv.Circle(image, (x,y), 0, color, 1)
    cv.ShowImage(window, image)


if __name__ == '__main__':
    stream = cv.CaptureFromFile(sys.argv[1])
    nFrames = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_COUNT))
    nRows = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_HEIGHT))
    nCols = int(cv.GetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_WIDTH))

    print "Resulution: %d x %d" % (nRows, nCols)
    print "Number of framees: %d" % nFrames

    cv.NamedWindow('video', 0)
    cv.NamedWindow('flow', 0)

    # for zero padding
    cur_frame_m = cv.CreateMat(nRows, nCols, cv.CV_8UC1)
    cv.Set(cur_frame_m, 0)
    pre_frame_m = cv.CreateMat(nRows, nCols, cv.CV_8UC1)
    cv.Set(pre_frame_m, 0)
    flow_m = cv.CreateMat(nRows, nCols, cv.CV_32FC2)
    cv.Set(flow_m, 0)

    for i in range(nFrames):
        # calculate the optical flow
        pre_frame_m, cur_frame_m = cur_frame_m, pre_frame_m
        frame = cv.QueryFrame(stream)
        cv.CvtColor(frame, cur_frame_m, cv.CV_BGR2GRAY)

        cv.CalcOpticalFlowFarneback(
            pre_frame_m, cur_frame_m, flow_m, pyr_scale=0.5, levels=3,
            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        cv.ShowImage('video', frame)
        draw_flow('flow', frame, flow_m)

        if(cv.WaitKey(100) == ('q' or 'Q')):
            break
