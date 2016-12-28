 #include <unistd.h>
#include "boost/algorithm/string.hpp"

#include "caffe/caffe.hpp"
#include "ristretto/quantization.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::string;
using caffe::vector;
using caffe::NetStateRule;
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

float my_pow(int q)
{
	#define 	START_Q 	(-31)	
	#define 	STOP_Q 		(31)	
	static float s_table_pow[STOP_Q-START_Q+10] = {0,};	
	static int init_flag =0;	
	if(init_flag == 0)	
	{		
		init_flag = 1;		
		for (int n = START_Q; n < STOP_Q; n++)	//定标		
		{			
			s_table_pow[n+(-START_Q)]  = pow((float)2, n);		
		}
	}
	if((q < START_Q)||(q >= STOP_Q))	
	{		
		printf(" Error ;-------------my_pow::: out of range-------------!!!\n");		
		assert(0);		
		return 1;
	}	
	else	
	{		
		return s_table_pow[q+(-START_Q)];	
	}
}

int DumpData2Txt(const char *filename,int width, int height, float*data)
{
	printf("DumpData2Txt : [%s] \n", filename);
	FILE *fp = fopen(filename, "w");
	if (fp == 0)
	{
		printf("DumpData2Txt ::: open file [%s] failed\n", filename);
		return 0;
	}
	fprintf(fp, "Witdh=%d,Height=%d !!!\n", width, height);
	for (int h = 0; h<height; h++)
	{
		for (int w = 0; w<width; w++)
		{
			fprintf(fp, "%10.6f,", data[h*width + w]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	fclose(fp);
	return 0;
}

Quantization::Quantization(string model, string weights, string model_quantized,
      int iterations, double error_margin, string gpus,string quantize_cfg,
      string debug_out_float,string debug_out_trim) {
	  this->model_ = model;
	  this->weights_ = weights;
	  this->model_quantized_ = model_quantized;
	  this->iterations_ = iterations;
	  this->error_margin_ = error_margin;
	  this->gpus_ = gpus;
	  this->quantize_cfg_ = quantize_cfg;
	  this->debug_out_float_ = debug_out_float;
	  this->debug_out_trim_ = debug_out_trim;
	  printf("[geyijun]---------Quantization:init---------\n");
	  printf("[geyijun]model=[%s]\n",model.c_str());
	  printf("[geyijun]weights=[%s]\n",weights.c_str());
	  printf("[geyijun]model_quantized=[%s]\n",model_quantized.c_str());
	  printf("[geyijun]iterations=[%d]\n",iterations);
	  printf("[geyijun]error_margin=[%f]\n",error_margin);
	  printf("[geyijun]quantize_cfg=[%s]\n",quantize_cfg.c_str());		
	  printf("[geyijun]debug_out_float=[%s]\n",debug_out_float.c_str());	
	  printf("[geyijun]debug_out_trim=[%s]\n",debug_out_trim.c_str());	
}

//geyijun@2016-12-09
//详细内容将学习笔记<CNN工程化问题>
//说明：数据统计也应该是在校验集上进行测试。
//因为训练集和校验集的层结构可能不同，所以都
//用校验集才能够统一起来。
void Quantization::QuantizeNet()
{
  	printf("[geyijun]---------Quantization:QuantizeNet---------start\n");
	CheckWritePermissions(model_quantized_+"./ttt.txt");
	SetGpu();

	//加载网络结构文件(获取网络层的名字)
	NetParameter param;
	caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
	param.mutable_state()->set_phase(caffe::TEST);
	for (int i = 0; i < param.layer_size(); ++i)
	{
		LayerParameter* param_layer = param.mutable_layer(i);
		bool layer_included = (param_layer->include_size() == 0);
		for (int j = 0; !layer_included && j < param_layer->include_size(); ++j) 
		{
			NetStateRule rule= param_layer->include(j);
			if(rule.has_phase() && (caffe::TEST == rule.phase()))
			{
				layer_included = true;;
			}	
      		}
		if(layer_included)
		{
			layer_names_.push_back(param_layer->name());
		}
	}
	for(int i=0;i<layer_names_.size();i++)
	{
		printf("[geyijun] layer_names_[%d] = [%s] \n",i,layer_names_[i].c_str());
	}

	//枚举所有的可用的bw  配置(要用layer_names_信息)
	ParseBwCfg();

	//挨个遍历所有的bw配置
	//再根据数据范围,进行局部最优搜索，定标.
	Net<float>*net_range = new Net<float>(param, NULL);
	net_range->CopyTrainedLayersFrom(weights_);
	CalcFlSign(10,net_range);
	delete net_range;
	sleep(5);

	//获取校验集上的基本分
	Net<float>*net_val = new Net<float>(param, NULL);
    	net_val->CopyTrainedLayersFrom(weights_);
	CalcBatchAccuracy(iterations_,net_val,&test_score_baseline_,NULL,0);
	if(!debug_out_float_.empty())//调试用
	{
		DumpAllBlobs2Txt(net_val,debug_out_float_);//最后一张图的状态	
	}
	delete net_val;
	printf("[geyijun] get--------->test_score_baseline_ = %f\n",test_score_baseline_);	
	sleep(5);

	//再在校验集上进行全局最优搜索.
	printf("[geyijun] try to search globol target\n");
	cfg_valid_data_bw_skip_.resize(cfg_valid_data_bw_.size());
	for(int i=0;i<cfg_valid_data_bw_.size();i++)
	{
		if(cfg_valid_data_bw_skip_[i] ==1)
		{
			continue;
		}
		printf("[geyijun] test cfg_valid_data_bw[%d]\n",i);
		NetParameter param;
		caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
		param.mutable_state()->set_phase(caffe::TEST);
		EditNetQuantizationParameter(&param,calc_params_bw_,calc_params_fl_,
								cfg_valid_data_bw_[i],calc_valid_data_fl_[i],
								calc_valid_data_sign_[i]);
    		Net<float>*net_val = new Net<float>(param, NULL);
    		net_val->CopyTrainedLayersFrom(weights_);

		float accuracy;
		CalcBatchAccuracy(iterations_,net_val,&accuracy,NULL,0);
		printf("[geyijun] get--------->accuracy = %f\n",accuracy);	
		if ( accuracy + error_margin_ / 100 > test_score_baseline_ ) 
		{
			//输出到文件中去
			char prototxt_name[128] = {0,};
			sprintf(prototxt_name,"%s/accuracy[%f]_cfg[%d].prototxt",model_quantized_.c_str(),accuracy,i);
			WriteProtoToTextFile(param, prototxt_name);
			printf("[geyijun] save it. --->[%s]\n",prototxt_name);
		}
		else
		{
			printf("[geyijun] skip it \n");	
			//遍历其后所有的配置
			for(int j=i+1;j<cfg_valid_data_bw_.size();j++)
			{
				if(CompareBwCfg(cfg_valid_data_bw_[i],cfg_valid_data_bw_[j])>0)
				{
					cfg_valid_data_bw_skip_[j] = 1;
				}
			}
		}
		if(!debug_out_trim_.empty()&&(i==0))//调试用
		{
			DumpAllBlobs2Txt(net_val,debug_out_trim_);//最后一张图的状态	
		}
		delete net_val;
	}
	printf("[geyijun]---------Quantization:QuantizeNet---------finish\n");
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
		if(cfg_auto_search_==1)
		{
			for(int bw=cfg_default_data_bw_;bw>=8;bw/=2)
			{
				valid_bw.push_back(bw);
			}
		}
		else
		{
			valid_bw.push_back(cfg_default_data_bw_);
		}
	}
	return valid_bw;
}

//枚举所有bw  配置的递归函数
vector<vector<int> >  Quantization::CombineBwCfg(vector< vector<int> > before_result,vector<int> cur_in)
{
	vector<vector<int> > after_result;
	after_result.clear();
	if(before_result.size() == 0)
	{
		for(int j=0;j<cur_in.size();j++)
		{
			vector<int> before_item;
			before_item.push_back(cur_in[j]);
			after_result.push_back(before_item);
		}
	}
	else
	{
		for(int i=0;i<before_result.size();i++)
		{
			for(int j=0;j<cur_in.size();j++)
			{
				vector<int> before_item = before_result[i];
				before_item.push_back(cur_in[j]);
				after_result.push_back(before_item);
			}
		}
	}
	return after_result;
}

//枚举出所有需要遍历的bw 组合
void Quantization::ParseBwCfg()
{
	printf("[geyijun] ParseBwCfg--------->1 \n");
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
	if(cfg_valid_data_bw_.size()> 8192)
	{
		LOG(FATAL) << "Too much valid data bw enum,please modify bw.cfg!"<<cfg_valid_data_bw_.size();
	}
	printf("[geyijun] cfg_conv_params_bw_=[%d]\n",cfg_conv_params_bw_);
	printf("[geyijun] cfg_ip_params_bw_=[%d]\n",cfg_ip_params_bw_);
	printf("[geyijun] cfg_default_data_bw_=[%d]\n",cfg_default_data_bw_);
	printf("[geyijun] cfg_auto_search_=[%d]\n",cfg_auto_search_);
	printf("[geyijun] cfg_valid_data_bw_.size=[%d]\n",(int)cfg_valid_data_bw_.size());
	/*
	for(int i=0;i<cfg_valid_data_bw_.size();i++)
	{
		printf("[geyijun] cfg_valid_data_bw_[%d]=",i);
		for(int j=0;j<cfg_valid_data_bw_[i].size();j++)
		{
			printf("<%02d>",cfg_valid_data_bw_[i][j]);
		}
		printf("\n");
	}
	*/
	printf("[geyijun] ParseBwCfg--------->end\n");
}

//计算量化能量差
float Quantization::Float2FixTruncate(float val,int bw,int fl,int is_sign)
{
	// Saturate data
	float max_data = 0;
	float min_data = 0;
	float result = 0;
	if(is_sign)
	{
		max_data = (my_pow(bw - 1) - 1) * my_pow(-fl);
		min_data = -my_pow(bw - 1) * my_pow(-fl);
	}
	else
	{
		max_data = (my_pow(bw) - 1) * my_pow(-fl);
		min_data = 0;
	}

	//对数据箝位一下
	result = std::max(std::min(val, max_data), min_data); 
	// Round data	
	result /= my_pow(-fl);	
	result = round(result);	
	result *= my_pow(-fl);	
	return result;	
}

float Quantization::CalcDataLoss(Blob<float>* blob,int bw,int fl,int is_sign)
{
	printf("[geyijun] CalcDataLoss--------->bw=[%d],fl=[%d],is_sign=[%d] \n",bw,fl,is_sign);
	float dataloss = 0;
	float* data = blob->mutable_cpu_data();
	int cnt = blob->count(); 
	for (int i = 0; i < cnt; ++i) 
	{
		dataloss += fabs(data[i] - Float2FixTruncate(data[i],bw,fl,is_sign));
	}
	return dataloss;
}

void Quantization::CalcFlSign(const int iterations,Net<float>* caffe_net)
{
	printf("[geyijun] CalcFlSign--------->1 \n");
	//先计算多次，统计各层的最大最小值
	for (int i = 0; i < iterations; ++i) 
	{
	       LOG(INFO) << "Running for " << iterations << " iterations."<<i<< " to get data range";
   		caffe_net->Forward();
		caffe_net->RangeInLayers(layer_names_,&max_params_, &max_data_, &min_data_);
	}
	printf("------------------------------------------------------------------------------------------------------\n");
	printf("------------------------------------------------------------------------------------------------------\n");
	printf("------------------------------------------------------------------------------------------------------\n");
	for(int i=0;i<layer_names_.size();i++)
	{
		printf("[geyijun] layer[%d][%s] --->max_params[%f] max_data[%f] min_data[%f]\n",i,layer_names_[i].c_str(),max_params_[i],max_data_[i],min_data_[i]);
	}
	printf("------------------------------------------------------------------------------------------------------\n");
	printf("------------------------------------------------------------------------------------------------------\n");
	printf("------------------------------------------------------------------------------------------------------\n");

	//先确定权值参数的数据定标
	printf("[geyijun] CalcFlSign--------->params's bw fl  \n");	
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
		printf("[geyijun] layer[%d][%s] --->calc_params_bw_[%d] calc_params_fl_[%d]\n",layer_id,layer_names_[layer_id].c_str(),calc_params_bw_[layer_id],calc_params_fl_[layer_id]);
	}
	printf("------------------------------------------------------------------------------------------------------\n");

	//在根据找到的最值决定Fl 和Sign标志(仅需要考虑data,不用考虑param)
	printf("[geyijun] CalcFlSign--------->data's bw fl  \n");				
	calc_valid_data_fl_.clear();							
	calc_valid_data_sign_.clear();			
	for (int i = 0; i < cfg_valid_data_bw_.size(); i++)
	{
		printf("------------------------------------------------------------------------------------------------------\n");
		printf("[geyijun] try to test cfg_valid_data_bw_[%d]=",i);
		for(int j=0;j<cfg_valid_data_bw_[i].size();j++)
		{
			printf("<%02d>",cfg_valid_data_bw_[i][j]);
		}
		printf("\n");

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
				Layer<float>* layer = caffe_net->layer_by_name(layer_names_[layer_id]).get();
			  	const LayerParameter& param_layer = layer->layer_param();
			    	const string& blob_name = param_layer.top(0);
				Blob<float>* blob = caffe_net->blob_by_name(blob_name).get();
				printf("[geyijun] layer_names[%s] blob_name[%s] ---> shape[%s]\n",layer_names_[layer_id].c_str(),blob_name.c_str(),blob->shape_string().c_str());

				int is_sign = (min_data_[layer_id]>=0)?0:1;
				int il  = (int)ceil(log2(max_data_[layer_id])+is_sign);	
				int fl_a = cfg_valid_data_bw_[i][layer_id]-il;	//定标
				int fl_b = fl_a+1;
				float  lost_a = CalcDataLoss(blob,cfg_valid_data_bw_[i][layer_id],fl_a,is_sign);
				float  lost_b = CalcDataLoss(blob,cfg_valid_data_bw_[i][layer_id],fl_b,is_sign);
				if(lost_a<lost_b)
				{
					fl_a_vote[layer_id]++;
				}
				else
				{
					fl_b_vote[layer_id]++;
				}
				printf("[geyijun] layer[%d][%s] --->lost_a[%f] lost_b[%f] ; a_vote[%d],b_vote[%d]\n",layer_id,layer_names_[layer_id].c_str(),lost_a,lost_b,fl_a_vote[layer_id],fl_b_vote[layer_id]);
			}			
		}
		vector<int> valid_data_sign;
		vector<int> valid_data_fl;
		valid_data_sign.resize(layer_names_.size());
		valid_data_fl.resize(layer_names_.size());
		for (int layer_id = 0; layer_id < layer_names_.size(); layer_id++)
		{
			int is_sign = (min_data_[layer_id]>=0)?0:1;
			valid_data_sign[layer_id] = is_sign;
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
		printf("------------------------------------------------------------------------------------------------------\n");
		printf("[geyijun] get valid_data_sign[%d]=",i);
		for(int j=0;j<valid_data_sign.size();j++)
		{
			printf("<%d>",valid_data_sign[j]);
		}
		printf("\n");
		printf("------------------------------------------------------------------------------------------------------\n");
		printf("[geyijun] get valid_data_fl[%d]=",i);
		for(int j=0;j<valid_data_fl.size();j++)
		{
			printf("<%d>",valid_data_fl[j]);
		}
		printf("\n");
		calc_valid_data_sign_.push_back(valid_data_sign);
		calc_valid_data_fl_.push_back(valid_data_fl);
		printf("------------------------------------------------------------------------------------------------------\n");
	}
	printf("[geyijun] CalcFlSign--------->end\n");	
}

//这个函数在校验集中用来最终误差
void Quantization::CalcBatchAccuracy(const int iterations,Net<float>* caffe_net,
										float* accuracy,float* cur_accuracy,const int score_number)
{
	printf("[geyijun] CalcBatchAccuracy--------->1\n");	
	LOG(INFO) << "Running for " << iterations << " iterations.";
	vector<int>	test_score_output_id;	
	vector<float>	test_score;	//所有输出network_out_blobs 的所有元素的记录(二维拉成一维)
	for (int i = 0; i < iterations; ++i) 
	{
   		//注意:::  只有LossLayer 层,才可以在Forward之后输出loss
		//我们看到如alexnet  ，无论是TRAIN  还是TEST 
		//最后一层都是SoftmaxWithLoss，所以可以获取loss. 就是和
		//label的误差。
		//所以针对不同网络这个地方的获取是有不同的，需要改代码

		//这个result  存放的是整个网络的所有的输出blobs(就是没有top link 的那些layer)
		//可以深入代码看看iter_loss  的获取过程。
		const vector<Blob<float>*>& result = caffe_net->Forward();

		//如果以alexnet 为例，这里
		//只有一个输出blobs 即result.size() == 1;
		//则一个输出blobs 只有一个元素即result[j]->count()=1
		// Keep track of network score over multiple batches.
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
				if((idx == score_number)&&(cur_accuracy))
				{
					*cur_accuracy = score;
				}
			}
		}
	}
	if(accuracy)
	{
		*accuracy = test_score[score_number] / iterations;
	}
	printf("[geyijun] CalcBatchAccuracy--------->end\n");	
}

void Quantization::EditNetQuantizationParameter(NetParameter* param,
							vector<int> params_bw,vector<int> params_fl,
							vector<int> data_bw,vector<int> data_fl,
							vector<int> data_sign)
{
	for (int i = 0; i < param->layer_size(); ++i)
	{
		LayerParameter* param_layer = param->mutable_layer(i);
		bool layer_included = (param_layer->include_size() == 0);
		for (int j = 0; !layer_included && j < param_layer->include_size(); ++j) 
		{
			NetStateRule rule= param_layer->include(j);
			if(rule.has_phase() && (caffe::TEST == rule.phase()))
			{
				layer_included = true;;
			}	
      		}
		if(layer_included == false)
		{
			continue;
		}
		const string& name = param_layer->name();
		vector<string>::iterator found_iter = find(layer_names_.begin(), layer_names_.end(),name); 
    		if ( found_iter == layer_names_.end( ) ) //没找到
    		{
    			continue;
    		}
		int found_index = found_iter - layer_names_.begin();
		const string& type_name = param_layer->type();
		//初步支持量化计算的层的种类(需要逐步增加)
		if((type_name == "InnerProduct")
		||(type_name == "Convolution")
		||(type_name == "Deconvolution")
		||(type_name == "LRN")
		||(type_name == "Pooling")
		||(type_name == "ReLU")
		||(type_name == "Data"))
		{
			string new_typename = type_name +"Ristretto";
			param_layer->set_type(new_typename);
		}
		param_layer->mutable_quantization_param()->set_bw_params(params_bw[found_index]);  
		param_layer->mutable_quantization_param()->set_fl_params(params_fl[found_index]);
		param_layer->mutable_quantization_param()->set_bw_layer_data(data_bw[found_index]);  
		param_layer->mutable_quantization_param()->set_fl_layer_data(data_fl[found_index]); 
		param_layer->mutable_quantization_param()->set_is_sign_data(data_sign[found_index]);
	}
}

//比较两组配置的精度大小
int Quantization::CompareBwCfg(vector<int>& bwcfg1,vector<int>& bwcfg2)
{
	assert(bwcfg1.size() == bwcfg2.size());
	for(int i=0;i<bwcfg1.size();i++)
	{
		if(bwcfg1[i] < bwcfg2[i])
		{
			return 0;
		}
	}
	return 1;
}

void Quantization::DumpAllBlobs2Txt(Net<float>* caffe_net,string dumpdir)
{
	for (int layer_id = 0; layer_id < layer_names_.size(); layer_id++)
	{
		Layer<float>* layer = caffe_net->layer_by_name(layer_names_[layer_id]).get();
		const LayerParameter& param_layer = layer->layer_param();
		const string& blob_name = param_layer.top(0);
		Blob<float>* blob = caffe_net->blob_by_name(blob_name).get();
		printf("[geyijun] blob_name[%s] shape[%s]\n",blob_name.c_str(),blob->shape_string().c_str());
		if(blob->num_axes() == 4)
		{
			for(int ch=0;ch<blob->shape(1);ch++)
			{
				char filename[128] = {0,};
				sprintf(filename,"%s/%s_%s_%d.txt",dumpdir.c_str(),layer_names_[layer_id].c_str(),blob_name.c_str(),ch);
				float* data = blob->mutable_cpu_data();
				int offset = blob->offset(0,ch); 
				DumpData2Txt(filename,blob->shape(3), blob->shape(2), data+offset);
			}
		}
		else if(blob->num_axes() == 2)
		{
			char filename[128] = {0,};
			sprintf(filename,"%s/%s_%s.txt",dumpdir.c_str(),layer_names_[layer_id].c_str(),blob_name.c_str());
			float* data = blob->mutable_cpu_data();
			DumpData2Txt(filename,blob->shape(0), blob->shape(1), data);
		}
		else if(blob->num_axes() == 0)	//标量
		{
			char filename[128] = {0,};
			sprintf(filename,"%s/%s_%s.txt",dumpdir.c_str(),layer_names_[layer_id].c_str(),blob_name.c_str());
			float* data = blob->mutable_cpu_data();
			DumpData2Txt(filename,1,1, data); 
		}
	}
}

