#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
SliceRistrettoLayer<Dtype>::SliceRistrettoLayer(const LayerParameter& param)
      : SliceLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() 
{
	this->bw_params_ = this->layer_param_.quantization_param().bw_params();
	this->fl_params_ = this->layer_param_.quantization_param().fl_params();
	this->bw_layer_out_= this->layer_param_.quantization_param().bw_layer_data();
	this->fl_layer_out_= this->layer_param_.quantization_param().fl_layer_data();
	this->is_sign_out = this->layer_param_.quantization_param().is_sign_data();
}

template <typename Dtype>
void SliceRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	//���ø����ǰ������
	SliceLayer<Dtype>::Forward_gpu(bottom,top);
	//�Խ�������������
	this->QuantizeLayerOutputs_cpu(top[0]->mutable_cpu_data(), top[0]->count());
}

template <typename Dtype>
void SliceRistrettoLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	return Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(SliceRistrettoLayer);
REGISTER_LAYER_CLASS(SliceRistretto);

}  // namespace caffe
