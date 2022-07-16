//
// Created by Asuka on 2022/6/18.
//

#ifndef CPP_PROJECT_MATRIXTOOPENCV_H
#define CPP_PROJECT_MATRIXTOOPENCV_H

#include "MyMatrix.h"
#include <opencv2/opencv.hpp>

cv::Mat parseToOpenCV(dense::DenseMat<uchar> &p) {
    cv::Mat a(cv::Size(p.col(), p.row()), CV_8UC1);
    for (int i = 0; i < p.row(); i++)
        for (int j = 0; j < p.col(); j++)
            a.at<uchar>(i, j) = (uchar) p.get(i + 1, j + 1);
    return a;
}

cv::Mat parseToOpenCV(dense::DenseMat<char> &p) {
    cv::Mat a(cv::Size(p.col(), p.row()), CV_8SC1);
    for (int i = 0; i < p.row(); i++)
        for (int j = 0; j < p.col(); j++)
            a.at<uchar>(i, j) = (char) p.get(i + 1, j + 1);
    return a;
}

cv::Mat parseToOpenCV(dense::DenseMat<ushort> &p) {
    cv::Mat a(cv::Size(p.col(), p.row()), CV_16UC1);
    for (int i = 0; i < p.row(); i++)
        for (int j = 0; j < p.col(); j++)
            a.at<ushort>(i, j) = (ushort) p.get(i + 1, j + 1);
    return a;
}

cv::Mat parseToOpenCV(dense::DenseMat<short> &p) {
    cv::Mat a(cv::Size(p.col(), p.row()), CV_16SC1);
    for (int i = 0; i < p.row(); i++)
        for (int j = 0; j < p.col(); j++)
            a.at<short>(i, j) = (short) p.get(i + 1, j + 1);
    return a;
}

cv::Mat parseToOpenCV(dense::DenseMat<int> &p) {
    cv::Mat a(cv::Size(p.col(), p.row()), CV_32SC1);
    for (int i = 0; i < p.row(); i++)
        for (int j = 0; j < p.col(); j++)
            a.at<int>(i, j) = (int) p.get(i + 1, j + 1);
    return a;
}

cv::Mat parseToOpenCV(dense::DenseMat<float> &p) {
    cv::Mat a(cv::Size(p.col(), p.row()), CV_32FC1);
    for (int i = 0; i < p.row(); i++)
        for (int j = 0; j < p.col(); j++)
            a.at<float>(i, j) = (float) p.get(i + 1, j + 1);
    return a;
}

cv::Mat parseToOpenCV(dense::DenseMat<double> &p) {
    cv::Mat a(cv::Size(p.col(), p.row()), CV_64FC1);
    for (int i = 0; i < p.row(); i++)
        for (int j = 0; j < p.col(); j++)
            a.at<double>(i, j) = (double) p.get(i + 1, j + 1);
    return a;
}

template<class T>
dense::DenseMat<T> parseToMatrix(cv::Mat &a) {
    if(a.type() > 6) throw invalid_argument("cv matrix is not single channel!");
    int n = a.rows, m = a.cols;
    dense::DenseMat<T> p(n, m);
    for (int i = 0; i < p.row(); i++)
        for (int j = 0; j < p.col(); j++) {
            int type = a.type();
            switch (type) {
                case CV_8UC1:
                    p.set(i + 1, j + 1, (T)a.at<uchar>(i, j));
                    break;
                case CV_8SC1:
                    p.set(i + 1, j + 1, (T)a.at<char>(i, j));
                    break;
                case CV_16UC1:
                    p.set(i + 1, j + 1, (T)a.at<ushort>(i, j));
                    break;
                case CV_16SC1:
                    p.set(i + 1, j + 1, (T)a.at<short>(i, j));
                    break;
                case CV_32SC1:
                    p.set(i + 1, j + 1, (T)a.at<int>(i, j));
                    break;
                case CV_32FC1:
                    p.set(i + 1, j + 1, (T)a.at<float>(i, j));
                    break;
                case CV_64FC1:
                    p.set(i + 1, j + 1, (T)a.at<double>(i, j));
                    break;
                default:
                    throw invalid_argument("cv matrix is not single channel!");
            }
        }
    return p;
}



#endif //CPP_PROJECT_MATRIXTOOPENCV_H
