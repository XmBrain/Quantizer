#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
PoolingRistrettoLayer<Dtype>::PoolingRistrettoLayer(const LayerParameter& param)
      : PoolingLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() 
{
	this->bw_params_ = this->layer_param_.quantization_param().bw_params();
	this->fl_params_ = this->layer_param_.quantization_param().fl_params();
	this->bw_layer_out_= this->layer_param_.quantization_param().bw_layer_data();
	this->fl_layer_out_= this->layer_param_.quantization_param().fl_layer_data();
	this->is_sign_out= this->layer_param_.quantization_param().is_sign_data();
}

template <typename Dtype>
void PoolingRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	//调用父类的前向运算
	PoolingLayer<Dtype>::Forward_cpu(bottom,top);
	//对结果进行量化输出
	this->QuantizeLayerOutputs_cpu(top[0]->mutable_cpu_data(), top[0]->count());
}

#ifdef CPU_ONLY
STUB_GPU(PoolingRistrettoLayer);
#endif

INSTANTIATE_CLASS(PoolingRistrettoLayer);
REGISTER_LAYER_CLASS(PoolingRistretto);

}  // namespace caffe
