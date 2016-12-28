#include <vector>

#include "ristretto/base_ristretto_layer.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
ConvolutionRistrettoLayer<Dtype>::ConvolutionRistrettoLayer(
	const LayerParameter& param) : ConvolutionLayer<Dtype>(param),
	BaseRistrettoLayer<Dtype>()
{
	this->bw_params_ = this->layer_param_.quantization_param().bw_params();
	this->fl_params_ = this->layer_param_.quantization_param().fl_params();
	this->bw_layer_out_= this->layer_param_.quantization_param().bw_layer_data();
	this->fl_layer_out_= this->layer_param_.quantization_param().fl_layer_data();
	this->is_sign_out= this->layer_param_.quantization_param().is_sign_data();
	printf("ConvolutionRistrettoLayer------------------------>111\n");
}

template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	//printf("ConvolutionRistrettoLayer------------------------>Forward_cpu\n");
	//对权值参数进行量化
	this->QuantizeWeights_cpu(this->blobs_, this->bias_term_);
	//调用父类的前向运算
	//ConvolutionLayer<Dtype>::Forward_cpu(bottom,top);
	ConvolutionLayer<Dtype>::Forward_gpu(bottom,top);
	//对结果进行量化输出
	this->QuantizeLayerOutputs_cpu(top[0]->mutable_cpu_data(), top[0]->count());
}

template <typename Dtype>
void ConvolutionRistrettoLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	return Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(ConvolutionRistrettoLayer);
REGISTER_LAYER_CLASS(ConvolutionRistretto);

}  // namespace caffe

