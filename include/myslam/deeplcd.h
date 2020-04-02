#ifndef MYSLAM_DEEPLCD_H
#define MYSLAM_DEEPLCD_H

#include "caffe/caffe.hpp"

#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>

#include <Eigen/Core>

#include <list>
#include <vector>
#include <iostream>


namespace myslam
{

// Deep Loop Closure Detector
class DeepLCD{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	typedef std::shared_ptr<DeepLCD> Ptr;
	
	typedef Eigen::Matrix<float, 1064, 1> DescrVector;

	caffe::Net<float>* autoencoder; // the deploy autoencoder
	caffe::Blob<float>* autoencoder_input; // The encoder's input blob
	caffe::Blob<float>* autoencoder_output; // The encoder's input blob

	// DeepLCD() {}
	// If gpu_id is -1, the cpu will be used
	DeepLCD(const std::string& network_definition_file="calc_model/deploy.prototxt", const std::string& pre_trained_model_file="calc_model/calc.caffemodel", int gpu_id=-1);
	
	~DeepLCD()
	{
		delete autoencoder;
	}


	const float score(const DescrVector& d1, const DescrVector& d2);

	DescrVector calcDescrOriginalImg(const cv::Mat& originalImg);
	const DescrVector calcDescr(const cv::Mat& im); // make a forward pass through the net, return the descriptor

private:
	
}; // end class DeepLCD


} // end namespace 

#endif
