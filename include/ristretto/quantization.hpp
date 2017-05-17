#ifndef QUANTIZATION_HPP_
#define QUANTIZATION_HPP_

#include "caffe/caffe.hpp"

using caffe::string;
using caffe::vector;
using caffe::Net;
using caffe::Blob;
using caffe::NetParameter;

/**
 * @brief Approximate 32-bit floating point networks.
 *
 * This is the Ristretto tool. Use it to generate file descriptions of networks
 * which use reduced word width arithmetic.
 */
class Quantization {
public:
  explicit Quantization(string model, string weights, string model_quantized,
      int iterations, double error_margin, string gpus,string quantize_cfg,
      string debug_out_float,string debug_out_trim);
  void QuantizeNet();
  //修改网络量化参数
  static void EditNetQuantizationParameter(NetParameter* param,
					  				vector<string> layer_names,
									vector<int> params_bw,vector<int> params_fl,
									vector<int> data_bw,vector<int> data_fl,
									vector<int> data_sign);
  
private:
  void CheckWritePermissions(const string path);
  void SetGpu();

  
  //枚举出所有需要遍历的bw 组合
  vector<int> GetValidBw(string cur_name);
  vector<vector<int> > CombineBwCfg(vector<vector<int> > before_result, vector<int> cur_in);
  void ParseBwCfg();

  //计算量化能量差
  float Float2FixTruncate(float val,int bw,int fl,int is_sign);		
  float CalcDataLoss(Blob<float>* blob,int bw,int fl,int is_sign);

  //根据一组bw配置,结合(最大值，最小值)，使用【局部最优】标准，决定该层的is_sign和fl参数。
  void CalcFlSign(const int iterations,Net<float>* caffe_net);
  void CalcFlSign_ForLSTM(Net<float>* caffe_net);
  
  //计算一个批次的精度
  void CalcBatchAccuracy(const int iterations,Net<float>* caffe_net, float* accuracy,float* cur_accuracy,const int score_number );

  //比较两组配置的精度大小
  int CompareBwCfg(vector<int>& bwcfg1,vector<int>& bwcfg2);
  void DumpAllBlobs2Txt(Net<float>* caffe_net,string dumpdir);
  
  //配置参数(用户输入的)
  string model_;
  string weights_;
  static string model_quantized_;
  int iterations_;
  double error_margin_;
  string gpus_;
  string quantize_cfg_;	//量化配置文件，用来指定每一层可用的bw 枚举值

  //调试输出用
  string debug_out_float_;
  string debug_out_trim_;
  
  //下面信息由bw.cfg 配置文件中获取到
  int cfg_conv_params_bw_; 	//用户配置的卷积层权重的位宽
  int cfg_ip_params_bw_;		//用户配置的全连接层权重的位宽
  int cfg_default_data_bw_;	//用户配置的默认的位宽
  int cfg_auto_search_;		//是否自动搜索最优  
  vector<vector<int> > cfg_valid_data_bw_;			//枚举出所有有效的bw配置(自动搜索用)
  vector<vector<int> > calc_valid_data_fl_;			//对应与上面的每一个bw 的fl 
  vector<vector<int> > calc_valid_data_sign_;		//对应与上面的每一个bw 的s_sign
  vector<float> max_params_, max_data_, min_data_;//统计所得的最值
  vector<int> cfg_valid_data_bw_skip_;	//忽略标志
 
  //注意:  权值参数的bw 直接根据用户配置的位宽来
  //固定使用有符号数，并且fl  直接根据max 计算就好了。
  vector<int> calc_params_bw_;
  vector<int> calc_params_fl_;
 
  //在训练集上float  运算的统计结果，用户局部最优定标
  vector<string> layer_names_;  //每一层的名字(注意和net中的有差别，net会多一些split层)
  
  //在检验集上float  运算的分数，作为全局评价的基准分
  float test_score_baseline_;	
};	

#endif // QUANTIZATION_HPP_
