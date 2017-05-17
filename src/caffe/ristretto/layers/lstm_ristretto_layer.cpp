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
}


template <typename Dtype>
void LSTMRistrettoLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const 
{
	printf("----------------------------LSTMRistrettoLayer FillUnrolledNet Enter---------------\n");
	//当前层的名字
	LayerParameter lstm_layer = Layer<Dtype>::layer_param();
	string lstm_layer_name = lstm_layer.name();
			  
	//调用父类的网络展开函数
	LSTMLayer<Dtype>::FillUnrolledNet(net_param);
	
	//加载量化文件
	string quantize_file = this->layer_param_.quantization_param().quantize_file(); 
	NetParameter quantize_net_param;
	ReadNetParamsFromTextFileOrDie(quantize_file, &quantize_net_param);

	//再把量化参数赋值给这个子网络
	for (int i = 0; i < net_param->layer_size(); ++i)
	{
		LayerParameter* param_layer = net_param->mutable_layer(i);
		string layer_name = param_layer->name();
		layer_name = lstm_layer_name+"_"+layer_name;
		int j = 0;
		for (j = 0; j < quantize_net_param.layer_size(); ++j)
		{
			LayerParameter* quantize_param_layer = quantize_net_param.mutable_layer(j);
			const string quantize_layer_name = quantize_param_layer->name();
			if(layer_name == quantize_layer_name)
			{
				printf("[DEBUG]---------------Set Quantize Param Layer[%s]\n",quantize_layer_name.c_str());
				if(NULL != strstr(quantize_param_layer->type().c_str(),"Ristretto"))
				{
					//修改类型
					const string type_name = param_layer->type();
					string new_typename = type_name +"Ristretto";
					param_layer->set_type(new_typename);
					//增加参数
					param_layer->mutable_quantization_param()->set_bw_params
						(quantize_param_layer->quantization_param().bw_params());  
					param_layer->mutable_quantization_param()->set_fl_params
						(quantize_param_layer->quantization_param().fl_params());  
					param_layer->mutable_quantization_param()->set_bw_layer_data
						(quantize_param_layer->quantization_param().bw_layer_data());  
					param_layer->mutable_quantization_param()->set_fl_layer_data
						(quantize_param_layer->quantization_param().fl_layer_data());  
					param_layer->mutable_quantization_param()->set_is_sign_data
						(quantize_param_layer->quantization_param().is_sign_data());  
				}
				
				break;
			}
		}
		if(j >= quantize_net_param.layer_size())
		{
			printf("[ERROR]---------------Not found match layer\n");
		}
	}
	printf("----------------------------LSTMRistrettoLayer FillUnrolledNet Exit---------------\n");
}

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
