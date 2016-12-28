#include <vector>

#include "ristretto/base_ristretto_layer.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
DeconvolutionRistrettoLayer<Dtype>::DeconvolutionRistrettoLayer(const LayerParameter& param) : DeconvolutionLayer<Dtype>(param),BaseRistrettoLayer<Dtype>() 
{
	this->bw_params_ = this->layer_param_.quantization_param().bw_params();
	this->fl_params_ = this->layer_param_.quantization_param().fl_params();
	this->bw_layer_out_= this->layer_param_.quantization_param().bw_layer_data();
	this->fl_layer_out_= this->layer_param_.quantization_param().fl_layer_data();
	this->is_sign_out= this->layer_param_.quantization_param().is_sign_data();
}

template <typename Dtype>
void DeconvolutionRistrettoLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	//调用父类的前向运算
	//DeconvolutionLayer<Dtype>::Forward_cpu(bottom,top);
	DeconvolutionLayer<Dtype>::Forward_gpu(bottom,top);
	//对结果进行量化输出
	this->QuantizeLayerOutputs_cpu(top[0]->mutable_cpu_data(), top[0]->count());
}

template <typename Dtype>
void DeconvolutionRistrettoLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	return Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(DeconvolutionRistrettoLayer);
REGISTER_LAYER_CLASS(DeconvolutionRistretto);

}  // namespace caffe
