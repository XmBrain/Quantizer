#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "ristretto/base_ristretto_layer.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
LSTMRistrettoLayer<Dtype>::LSTMRistrettoLayer(const LayerParameter& param)
      : LSTMLayer<Dtype>(param), BaseRistrettoLayer<Dtype>() 
{
	this->bw_params_ = this->layer_param_.quantization_param().bw_params();
	this->fl_params_ = this->layer_param_.quantization_param().fl_params();
	this->bw_layer_out_= this->layer_param_.quantization_param().bw_layer_data();
	this->fl_layer_out_= this->layer_param_.quantization_param().fl_layer_data();
	this->is_sign_out = this->layer_param_.quantization_param().is_sign_data();

	//加载量化文件
	quantize_file_ = this->layer_param_.quantization_param().quantize_file(); 
	ReadNetParamsFromTextFileOrDie(quantize_file_, &quantize_net_param_);
	quantize_net_param_.mutable_state()->set_phase(caffe::TEST);
}

/*
template <typename Dtype>
void LSTMRistrettoLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const 
{
	//量化后子网络参数
	*net_param = quantize_net_param_;
}
*/

template <typename Dtype>
void LSTMRistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	//调用父类的前向运算
	LSTMLayer<Dtype>::Forward_gpu(bottom,top);
	//对结果进行量化输出
	this->QuantizeLayerOutputs_cpu(top[0]->mutable_cpu_data(), top[0]->count());
}

template <typename Dtype>
void LSTMRistrettoLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	return Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(LSTMRistrettoLayer);
REGISTER_LAYER_CLASS(LSTMRistretto);

}  // namespace caffe
