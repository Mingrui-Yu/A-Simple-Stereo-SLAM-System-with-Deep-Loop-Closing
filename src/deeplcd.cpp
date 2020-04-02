/**
 *  This file is modified from "deeplcd.cpp" in calc/DeepLCD: https://github.com/rpng/calc
 */

#include "myslam/deeplcd.h"
#include <opencv2/core.hpp>

namespace myslam{

DeepLCD::DeepLCD(const std::string& network_definition_file, 
	const std::string& pre_trained_model_file,
	int gpu_id){

	std::string mode = "CPU";
	if(gpu_id >= 0){
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		caffe::Caffe::SetDevice(gpu_id);
		mode = "GPU";
	}
	else {
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	}
	clock_t begin = clock();
	autoencoder = new caffe::Net<float>(network_definition_file, caffe::TEST);	
	autoencoder->CopyTrainedLayersFrom(pre_trained_model_file);
	clock_t end = clock();
	std::cout << "\nCaffe mode = " << mode << "\n";
	std::cout << "Loaded model in " <<   double(end - begin) / CLOCKS_PER_SEC << " seconds\n";
	autoencoder_input = autoencoder->input_blobs()[0]; // construct the input blob shared_ptr
	autoencoder_output = autoencoder->output_blobs()[0]; // construct the output blob shared_ptr
}

// ----------------------------------------------------------------------------------------

const float DeepLCD::score(const DescrVector& d1, const DescrVector& d2)
{
	float result = d1.transpose() * d2;
	return result;
}


// ----------------------------------------------------------------------------------------
DeepLCD::DescrVector DeepLCD::calcDescrOriginalImg(const cv::Mat& originalImg){
	assert(!originalImg.empty());

	cv::GaussianBlur(originalImg, originalImg, cv::Size(7, 7), 0);

	cv::Size _sz(160, 120);
	cv::Mat imResize;
	cv::resize(originalImg, imResize, _sz);
	return calcDescr(imResize);
}
// ----------------------------------------------------------------------------------------

const DeepLCD::DescrVector DeepLCD::calcDescr(const cv::Mat& im_){
	// the input image needs to be resized before
	
	std::vector<cv::Mat> input_channels(1); //We need this wrapper to place data into the net. Allocate space for at most 3 channels	
	int w = autoencoder_input->width();
	int h = autoencoder_input->height();
	float* input_data = autoencoder_input->mutable_cpu_data();
	cv::Mat channel(h, w, CV_32FC1, input_data);
	input_channels.emplace(input_channels.begin(), channel);
	input_data += w * h;
	cv::Mat im(im_.size(), CV_32FC1);
	im_.convertTo(im, CV_32FC1, 1.0/255.0); // convert to [0,1] grayscale. Place in im instead of im_
	// This will write the image to the input layer of the net
	cv::split(im, input_channels);
	autoencoder->Forward(); // Calculate the forward pass
	const float* tmp_descr;
	tmp_descr = autoencoder_output->cpu_data(); 
	int p = autoencoder_output->channels(); // Flattened layer get the major axis in channels dimension

	// We need to copy the data, or it will be overwritten on the next Forward() call
	// We may have a TON of desciptors, so allocate on the heap to avoid stack overflow
	int sz = p * sizeof(float);
	float* descr_ = (float*)std::malloc(sz);
	std::memcpy(descr_, tmp_descr, sz);

	assert(p == 1064);

	DescrVector descriptor;
	for(int i = 0; i < p; i++){
		descriptor(i, 0) = *(descr_ + i);
	}

	// normalization
	descriptor /= descriptor.norm();

	return descriptor;
}
	

} // end namespace















