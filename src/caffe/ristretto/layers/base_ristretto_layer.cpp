#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
BaseRistrettoLayer<Dtype>::BaseRistrettoLayer() 
{
	;
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerOutputs_cpu(Dtype* data, const int count) 
{
	Trim2FixedPoint_cpu(data, count, bw_layer_out_, fl_layer_out_, is_sign_out);
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeWeights_cpu(
	vector<shared_ptr<Blob<Dtype> > > weights_quantized,const bool bias_term) 
{
	Trim2FixedPoint_cpu(weights_quantized[0]->mutable_cpu_data(),
					weights_quantized[0]->count(),
					bw_params_, fl_params_,1);
	if (bias_term) 
	{
		Trim2FixedPoint_cpu(weights_quantized[1]->mutable_cpu_data(),
						weights_quantized[1]->count(), 
						bw_params_, fl_params_, 1);
	}
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width,int fl,int is_sign)
{
	// Saturate data
	Dtype max_data = 0;
	Dtype min_data = 0;
	if(is_sign)
	{
		max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
		min_data = -pow(2, bit_width - 1) * pow(2, -fl);
	}
	else
	{
		max_data = (pow(2, bit_width) - 1) * pow(2, -fl);
		min_data = 0;
	}

	for (int index = 0; index < cnt; ++index) 
	{
		data[index] = std::max(std::min(data[index], max_data), min_data); //对数据箝位一下
		// Round data
		data[index] /= pow(2, -fl);			//
		data[index] = round(data[index]);	//
		data[index] *= pow(2, -fl);			//
	}
}

//这里是要构造这些模板函数
template BaseRistrettoLayer<double>::BaseRistrettoLayer();
template BaseRistrettoLayer<float>::BaseRistrettoLayer();
template void BaseRistrettoLayer<double>::QuantizeWeights_cpu(vector<shared_ptr<Blob<double> > > weights_quantized,const bool bias_term);
template void BaseRistrettoLayer<float>::QuantizeWeights_cpu(vector<shared_ptr<Blob<float> > > weights_quantized,const bool bias_term);
template void BaseRistrettoLayer<double>::QuantizeLayerOutputs_cpu(double* data,const int count);
template void BaseRistrettoLayer<float>::QuantizeLayerOutputs_cpu(float* data,const int count);
template void BaseRistrettoLayer<double>::Trim2FixedPoint_cpu(double* data,const int cnt, const int bit_width,int fl,int is_sign);
template void BaseRistrettoLayer<float>::Trim2FixedPoint_cpu(float* data,const int cnt, const int bit_width,int fl,int is_sign);

}  // namespace caffe

