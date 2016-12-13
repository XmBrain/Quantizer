#include "boost/algorithm/string.hpp"

#include "caffe/caffe.hpp"
#include "ristretto/quantization.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::string;
using caffe::vector;
using caffe::Blob;
using caffe::LayerParameter;
using caffe::NetParameter;

//读取bw.cfg 配置文件用的
int 	GetStringValue(const char *filename, const char *str, const char separate, char *value, char *raw)
{
	FILE *fp;
	char line[128];
	char tmpstr[128];
	char *pos;
	if (strlen(str) > 127)
	{
		printf("<ERROR>: strlen=%d\n", (int)strlen(str));
		return -1;
	}
	if (NULL == (fp = fopen(filename, "r")))
	{
		printf("<ERROR>:open %s failed\n", filename);
		return -1;
	}
	while(fgets(line, sizeof(line), fp)) /*lager than BYTES_PER_LINE is bug*/
	{
		pos = strstr(line, str);
		if (NULL != pos)	
		{	
			sprintf(tmpstr, "%s%c", str, separate);
			pos += strlen(tmpstr);	
			if (NULL != raw){
				if (strlen(pos) > 128)
					strncpy(raw, pos,128);
				else
					strncpy(raw, pos, strlen(pos));
			}	
			if (NULL != value)
				sscanf(pos, "%s", value);
			fclose(fp);
			return 0;
		}
	}
	fclose(fp);
	return -1;
}

Quantization::Quantization(string model, string weights, string model_quantized,
      int iterations, double error_margin, string gpus,string quantize_cfg) {
  this->model_ = model;
  this->weights_ = weights;
  this->model_quantized_ = model_quantized;
  this->iterations_ = iterations;
  this->error_margin_ = error_margin;
  this->gpus_ = gpus;
  this->quantize_cfg_ = quantize_cfg;
}

//geyijun@2016-12-09
//详细内容将学习笔记<CNN工程化问题>
void Quantization::QuantizeNet() 
{
	CheckWritePermissions(model_quantized_);
	SetGpu();

	//创建训练集
	Net<float>* net_train = new Net<float>(model_, caffe::TRAIN);
	net_train->CopyTrainedLayersFrom(weights_);
	layer_names_ = net_train->layer_names();

	//枚举所有的可用的bw  配置(要用layer_names_信息)
	ParseBwCfg();

	//挨个遍历所有的bw配置
	//再根据数据范围,在训练集上进行局部最优搜索，定标.
	CalcFlSign(100,net_train);
	delete net_train;

	//再在校验集上进行全局最优搜索.
	for(int i=0;i<cfg_valid_data_bw_.size();i++)
	{
		NetParameter param;
		caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
		param.mutable_state()->set_phase(caffe::TEST);
		EditNetQuantizationParameter(&param,calc_params_bw_,calc_params_fl_,
								cfg_valid_data_bw_[i],calc_valid_data_fl_[i],
								calc_valid_data_sign_[i]);
    		Net<float>*net_val = new Net<float>(param, NULL);
    		net_val->CopyTrainedLayersFrom(weights_);

		float accuracy;
		CalcBatchAccuracy(iterations_,net_val,&accuracy,0);
		if ( accuracy + error_margin_ / 100 < test_score_baseline_ ) 
		{
			//输出到文件中去
			char prototxt_name[128] = {0,};
			sprintf(prototxt_name,"accuracy[%f]_cfg[%d]_%s",accuracy,i,model_quantized_.c_str());
			WriteProtoToTextFile(param, prototxt_name);			
		}
		delete net_val;
	}
	return ;
}

void Quantization::CheckWritePermissions(const string path) 
{
	std::ofstream probe_ofs(path.c_str());
	if (probe_ofs.good()) 
	{
		probe_ofs.close();
		std::remove(path.c_str());
	} 
	else 
	{
		LOG(FATAL) << "Missing write permissions";
	}
}

void Quantization::SetGpu() {
	// Parse GPU ids or use all available devices
	vector<int> gpus;
	if (gpus_ == "all") 
	{
		int count = 0;
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDeviceCount(&count));
#else
		NO_GPU;
#endif
		for (int i = 0; i < count; ++i) 
		{
			gpus.push_back(i);
		}
	} 
	else if (gpus_.size()) 
	{
		vector<string> strings;
		boost::split(strings, gpus_, boost::is_any_of(","));
		for (int i = 0; i < strings.size(); ++i) 
		{
			gpus.push_back(boost::lexical_cast<int>(strings[i]));
		}
	} 
	else 
	{
		CHECK_EQ(gpus.size(), 0);
	}
	// Set device id and mode
	if (gpus.size() != 0) 
	{
		LOG(INFO) << "Use GPU with device ID " << gpus[0];
		Caffe::SetDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
	} 
	else 
	{
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
}

//当前层有效的bw 值
vector<int> Quantization::GetValidBw(string cur_name)
{
	vector<int> valid_bw;
	valid_bw.clear();
	
	char tmp[32] = {0,};
	memset(tmp,0,sizeof(tmp));
	if (GetStringValue(quantize_cfg_.c_str(), cur_name.c_str(), '=', tmp, NULL) >= 0)
	{
		int bw = atoi(tmp);
		valid_bw.push_back(bw);
	}
	else
	{	
		for(int bw=cfg_default_data_bw_;(cfg_auto_search_==1)&&(bw>=8);bw/=2)
		{
			valid_bw.push_back(bw);
		}
	}
	return valid_bw;
}

//枚举所有bw  配置的递归函数
vector<vector<int> >  Quantization::CombineBwCfg(vector< vector<int> > before_result,vector<int> cur_in)
{
	vector<vector<int> > after_result;
	after_result.clear();
	for(int i=0;i<before_result.size();i++)
	{
		vector<int> before_item = before_result[i];
		for(int j=0;j<cur_in.size();j++)
		{
			before_item.push_back(cur_in[j]);
			after_result.push_back(before_item);
		}
	}
	return after_result;
}

//枚举出所有需要遍历的bw 组合
void Quantization::ParseBwCfg()
{
	char tmp[32] = {0,};
	memset(tmp,0,sizeof(tmp));
	if (GetStringValue(quantize_cfg_.c_str(), "conv_weight_bw", '=', tmp, NULL) < 0)
	{
		LOG(FATAL) << "Missing conv_weight_bw in bw.cfg";
	}
	cfg_conv_params_bw_ = atoi(tmp);

	memset(tmp,0,sizeof(tmp));
	if (GetStringValue(quantize_cfg_.c_str(), "ip_weight_bw", '=', tmp, NULL) < 0)
	{
		LOG(FATAL) << "Missing ip_weight_bw in bw.cfg";
	}
	cfg_ip_params_bw_ = atoi(tmp);

	memset(tmp,0,sizeof(tmp));
	if (GetStringValue(quantize_cfg_.c_str(), "default_data_bw", '=', tmp, NULL) < 0)
	{
		LOG(FATAL) << "Missing default_data_bw in bw.cfg";
	}
	cfg_default_data_bw_ = atoi(tmp);

	cfg_auto_search_ = 0;
	memset(tmp,0,sizeof(tmp));
	if (GetStringValue(quantize_cfg_.c_str(), "auto_search", '=', tmp, NULL) >= 0)
	{
		cfg_auto_search_ = atoi(tmp);
	}

	//枚举所有可能的组合
	cfg_valid_data_bw_.clear();
	for (int i = 0; i < layer_names_.size(); i++)
	{
		vector<int> cur_in = GetValidBw(layer_names_[i]);
		cfg_valid_data_bw_ = CombineBwCfg(cfg_valid_data_bw_,cur_in);
  	}
	if(cfg_valid_data_bw_.size()> 1024)
	{
		LOG(FATAL) << "Too much valid data bw enum,please modify bw.cfg!";
	}
}

void Quantization::CalcFlSign(const int iterations,Net<float>* caffe_net)
{
	//先计算多次，统计各层的最大最小值
	for (int i = 0; i < iterations; ++i) 
	{
	       LOG(INFO) << "Running for " << iterations << " iterations. to get data range";
   		caffe_net->Forward();
		caffe_net->RangeInLayers(&max_params_, &max_data_, &min_data_);
	}

	//先确定权值参数的数据定标
        for (int layer_id = 0; layer_id < layer_names_.size(); layer_id++)
	{
		calc_params_bw_.push_back(0);
		calc_params_fl_.push_back(0);
		Layer<float>* layer = caffe_net->layer_by_name(layer_names_[layer_id]).get();
		if (strcmp(layer->type(), "Convolution") == 0 )
		{
			int il  = (int)ceil(log2(max_params_[layer_id])+1);	
			int fl = cfg_conv_params_bw_-il;
			calc_params_bw_[layer_id] = cfg_conv_params_bw_;
			calc_params_fl_[layer_id] = fl;
		}
		else if (strcmp(layer->type(), "InnerProduct") == 0 )
		{
			int il  = (int)ceil(log2(max_params_[layer_id])+1);	
			int fl = cfg_ip_params_bw_-il;
			calc_params_bw_[layer_id] = cfg_ip_params_bw_;
			calc_params_fl_[layer_id] = fl;
		}
	}
	
	//在根据找到的最值决定Fl 和Sign标志(仅需要考虑data,不用考虑param)
	calc_valid_data_fl_.clear();	
	calc_valid_data_sign_.clear();
	for (int i = 0; i < cfg_valid_data_bw_.size(); i++)
	{
		vector<int> fl_a_vote;	//投票给fl_a  的个数
		vector<int> fl_b_vote;	//投票给fl_b  的个数
		fl_a_vote.resize(layer_names_.size());
		fl_b_vote.resize(layer_names_.size());
		for (int iter = 0; iter < iterations; iter++) 
		{
			LOG(INFO) << "Running for " << iterations << " iterations. to get fl and is_sign : "<< iter;
			caffe_net->Forward();  
			for (int layer_id = 0; layer_id < layer_names_.size(); layer_id++)
			{
				int is_sign = (min_data_[layer_id]>=0)?0:1;
				int il  = (int)ceil(log2(max_data_[layer_id])+is_sign);	
				int fl_a = cfg_valid_data_bw_[i][layer_id]-il;	//定标
				int fl_b = fl_a-1;

				float  lost_a = caffe_net->CalcDataLoss(layer_id,cfg_valid_data_bw_[i][layer_id],fl_a,is_sign);
				float  lost_b = caffe_net->CalcDataLoss(layer_id,cfg_valid_data_bw_[i][layer_id],fl_b,is_sign);
				if(lost_a<lost_b)
				{
					fl_a_vote[layer_id]++;
				}
				else
				{
					fl_b_vote[layer_id]++;
				}
			}			
		}
		vector<int> valid_data_sign;
		vector<int> valid_data_fl;
		valid_data_sign.resize(layer_names_.size());
		valid_data_fl.resize(layer_names_.size());
		for (int layer_id = 0; layer_id < layer_names_.size(); layer_id++)
		{
			int is_sign = (min_data_[layer_id]>=0)?0:1;
			int il  = (int)ceil(log2(max_data_[layer_id])+is_sign);	
			int fl_a = cfg_valid_data_bw_[i][layer_id]-il;
			int fl_b = fl_a-1;
			if(fl_a_vote[layer_id] >= fl_a_vote[layer_id])
			{
				valid_data_fl[layer_id] = fl_a;
			}
			else
			{
				valid_data_fl[layer_id] = fl_b;
			}
		}
		calc_valid_data_sign_.push_back(valid_data_sign);
		calc_valid_data_fl_.push_back(valid_data_fl);
	}
}

//这个函数在校验集中用来最终误差
void Quantization::CalcBatchAccuracy(const int iterations,Net<float>* caffe_net,
										float* accuracy,const int score_number)
{
	LOG(INFO) << "Running for " << iterations << " iterations.";
	vector<int>	test_score_output_id;	
	vector<float>	test_score;	//所有输出network_out_blobs 的所有元素的记录(二维拉成一维)
	float loss = 0;	//总误差

	assert(accuracy);		
	for (int i = 0; i < iterations; ++i) 
	{
   		 float iter_loss;	//当次误差

		//注意:::  只有LossLayer 层,才可以在Forward之后输出loss
		//我们看到如alexnet  ，无论是TRAIN  还是TEST 
		//最后一层都是SoftmaxWithLoss，所以可以获取loss. 就是和
		//label的误差。
		//所以针对不同网络这个地方的获取是有不同的，需要改代码

		//这个result  存放的是整个网络的所有的输出blobs(就是没有top link 的那些layer)
		//可以深入代码看看iter_loss  的获取过程。
		const vector<Blob<float>*>& result = caffe_net->Forward(&iter_loss);

		//如果以alexnet 为例，这里
		//只有一个输出blobs 即result.size() == 1;
		//则一个输出blobs 只有一个元素即result[j]->count()=1
		// Keep track of network score over multiple batches.
		loss += iter_loss;
		int idx = 0;	//把二维拉成一维的索引
		for (int j = 0; j < result.size(); ++j) 
		{
			const float* result_vec = result[j]->cpu_data();
			for (int k = 0; k < result[j]->count(); ++k, ++idx) 
			{
				const float score = result_vec[k];
				if (i == 0) 
				{
					test_score.push_back(score);
					test_score_output_id.push_back(j);
				}
				else
				{
					test_score[idx] += score;
				}
				const std::string& output_name = caffe_net->blob_names()[
										caffe_net->output_blob_indices()[j]];
				LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
			}
		}
	}
	loss /= iterations;
	LOG(INFO) << "Loss: " << loss;
	for (int i = 0; i < test_score.size(); ++i) 
	{
		const std::string& output_name = caffe_net->blob_names()[
									caffe_net->output_blob_indices()[test_score_output_id[i]]];
		const float loss_weight = caffe_net->blob_loss_weights()[
									caffe_net->output_blob_indices()[test_score_output_id[i]]];
		std::ostringstream loss_msg_stream;
		const float mean_score = test_score[i] / iterations;
		if (loss_weight) 
		{
			loss_msg_stream << " (* " << loss_weight << " = " << loss_weight * mean_score << " loss)";
		}
		LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
	}
	*accuracy = test_score[score_number] / iterations;
}

void Quantization::EditNetQuantizationParameter(NetParameter* param,
							vector<int> params_bw,vector<int> params_fl,
							vector<int> data_bw,vector<int> data_fl,
							vector<int> data_sign)
{
	for (int i = 0; i < param->layer_size(); ++i)
	{
		LayerParameter* param_layer = param->mutable_layer(i);
		const string& type_name = param_layer->type();
		//初步支持量化计算的层的种类(需要逐步增加)
		if((type_name == "InnerProduct")
		||(type_name == "Convolution"))
		{
			string new_typename = type_name +"Ristretto";
			param_layer->set_type(new_typename);
		}
		param_layer->mutable_quantization_param()->set_bw_params(params_bw[i]);  
		param_layer->mutable_quantization_param()->set_fl_params(params_fl[i]);
		param_layer->mutable_quantization_param()->set_bw_layer_data(data_bw[i]);  
		param_layer->mutable_quantization_param()->set_fl_layer_data(data_fl[i]); 
		param_layer->mutable_quantization_param()->set_is_sign_data(data_sign[i]);
	}
}

