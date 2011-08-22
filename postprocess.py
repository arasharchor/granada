#!/usr/bin/env python2.7

import os
import os.path as path

import numpy
import svmutil
from  numpy import linalg

def get_confusion_matrix(ground, predict, labels):
    assert(ground.shape == predict.shape)

    conf_matrix = numpy.zeros((len(labels), len(labels)))
    for gi, gv in enumerate(labels):
        for pi, pv in enumerate(labels):
            conf_matrix[gi, pi] = (predict[ground == gv] == pv).sum()

    conf_matrix = conf_matrix / conf_matrix.sum(1).reshape((-1, 1))
    return conf_matrix


if __name__ == "__main__":

    flow_path = '/home/zhanwu/storage/KTH/flow'

    action_names = ['boxing', 'handclapping', 'handwaving', 'jogging',
                    'running', 'walking']

    # number of training and testing samples
    n_train = 16
    n_test = 9
    d_feature = 45

    # read training and testing feature
    train_feature = numpy.empty((0, d_feature), 'float32')
    train_label = numpy.empty((0, 3), 'int16')

    test_feature = numpy.empty((0, d_feature), 'float32')
    test_label = numpy.empty((0, 3), 'int16')

    # load features into train/test container
    for idx, action in enumerate(action_names):
        flow_action_path = path.join(flow_path, action)
        file_names = os.listdir(flow_action_path)
        for file_name in file_names:
            file_path = path.join(flow_action_path, file_name)
            feature = numpy.load(file_path)
            # # normalize the feature
            # feature = feature/linalg.norm(feature)
            feature = feature.transpose()
            feature = numpy.log(numpy.abs(feature))

            person_id = int(file_name[6:8])
            env_id = int(file_name[-15])
            label = numpy.array([idx, person_id, env_id])

            if person_id <= n_train:
                train_feature = numpy.vstack((train_feature, feature))
                train_label = numpy.vstack((train_label, label))
            else:
                test_feature = numpy.vstack((test_feature, feature))
                test_label = numpy.vstack((test_label, label))

    # prepare data in libsvm favored format
    train_feature_svm = []
    for row in xrange(train_feature.shape[0]):
        train_feature_svm.append(dict(zip(range(1, d_feature + 1), train_feature[row, ])))

    test_feature_svm = []
    for row in xrange(test_feature.shape[0]):
        test_feature_svm.append(dict(zip(range(1, d_feature + 1), test_feature[row, ])))

    # train svm models
    models = []
    for n_action in xrange(len(action_names)):
        feature = train_feature_svm

        label = numpy.ones((train_label.shape[0]))
        label[train_label[:, 0] != n_action] = -1
        # to make sure that the label of the first example is 1
        label = label * label[0];
        label = list(label)

        # -t 0: linear svm
        model = svmutil.svm_train(label, feature, '-t 0 -b 1 -h 0')
        models.append(model)
        print 'finished training class: %d' % n_action

    # test svm models
    result = []
    for n_action in xrange(len(action_names)):
        feature = test_feature_svm

        label = numpy.ones((test_label.shape[0]))
        label[test_label[:, 0] != n_action] = -1
        label = list(label)

        _, _, p = svmutil.svm_predict(label, feature, models[n_action], '-b 1')
        result.append(p)

    result = numpy.array(result)
    positive_possibility = result[:, :, 0]
    positive_possibility[1:7, :] = 1 - positive_possibility[1:7, :]

    predict_label = numpy.argmax(positive_possibility, 0)
    p_label = predict_label.copy()

    begin = 0
    end = begin + 1
    while end < p_label.shape[0]:
        if ((test_label[begin, 1] == test_label[end, 1]) and
            (test_label[begin, 2] == test_label[end, 2])):
            end += 1
        else:
            lst = list(p_label[begin:end])
            ml = max(set(lst), key=lst.count)
            p_label[begin:end] = ml
            begin = end
            end = begin + 1
    lst = list(p_label[begin:end])
    ml = max(set(lst), key=lst.count)
    p_label[begin:end] = ml

    ground_label = test_label[:, 0]
    conf_matrix = get_confusion_matrix(ground_label, predict_label, range(6))
    print conf_matrix

    conf_matrix2 = get_confusion_matrix(ground_label, p_label, range(6))
    print conf_matrix2
