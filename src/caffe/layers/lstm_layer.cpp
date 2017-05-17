#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "ristretto/quantization.hpp"
namespace caffe {

template <typename Dtype>
void LSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}

template <typename Dtype>
void LSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_" + format_int(this->T_);
  (*names)[1] = "c_T";
}

template <typename Dtype>
void LSTMLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  const int num_blobs = 2;
  shapes->resize(num_blobs);
  for (int i = 0; i < num_blobs; ++i) {
    (*shapes)[i].Clear();
    (*shapes)[i].add_dim(1);  // a single timestep
    (*shapes)[i].add_dim(this->N_);
    (*shapes)[i].add_dim(num_output);
  }
}

template <typename Dtype>
void LSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h";
}

template <typename Dtype>
void LSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(num_output * 4);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->set_axis(2);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter scale_param;
  scale_param.set_type("Scale");
  scale_param.mutable_scale_param()->set_axis(0);

  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);

  LayerParameter split_param;
  split_param.set_type("Split");

  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ(2, input_shapes.size());

  LayerParameter* input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");
  input_layer_param->set_name("Input_c0h0");
  InputParameter* input_param = input_layer_param->mutable_input_param();

  input_layer_param->add_top("c_0");
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  input_layer_param->add_top("h_0");
  input_param->add_shape()->CopyFrom(input_shapes[1]);

  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(0);

  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xc_x = W_xc * x + b_c
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xc");
    x_transform_param->add_param()->set_name("b_c");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xc_x");
    x_transform_param->add_propagate_down(true);
  }

  if (this->static_input_) {
    // Add layer to transform x_static to the gate dimension.
    //     W_xc_x_static = W_xc_static * x_static
    LayerParameter* x_static_transform_param = net_param->add_layer();
    x_static_transform_param->CopyFrom(hidden_param);
    x_static_transform_param->mutable_inner_product_param()->set_axis(1);
    x_static_transform_param->set_name("W_xc_x_static");
    x_static_transform_param->add_param()->set_name("W_xc_static");
    x_static_transform_param->add_bottom("x_static");
    x_static_transform_param->add_top("W_xc_x_static_preshape");
    x_static_transform_param->add_propagate_down(true);

    LayerParameter* reshape_param = net_param->add_layer();
    reshape_param->set_type("Reshape");
    BlobShape* new_shape =
         reshape_param->mutable_reshape_param()->mutable_shape();
    new_shape->add_dim(1);  // One timestep.
    // Should infer this->N as the dimension so we can reshape on batch size.
    new_shape->add_dim(-1);
    new_shape->add_dim(
        x_static_transform_param->inner_product_param().num_output());
    reshape_param->set_name("W_xc_x_static_reshape");
    reshape_param->add_bottom("W_xc_x_static_preshape");
    reshape_param->add_top("W_xc_x_static");
  }

  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom("W_xc_x");
  x_slice_param->set_name("W_xc_x_slice");

  LayerParameter output_concat_layer;
  output_concat_layer.set_name("h_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("h");
  output_concat_layer.mutable_concat_param()->set_axis(0);

  for (int t = 1; t <= this->T_; ++t) {
    string tm1s = format_int(t - 1);
    string ts = format_int(t);

    cont_slice_param->add_top("cont_" + ts);
    x_slice_param->add_top("W_xc_x_" + ts);

    // Add layers to flush the hidden state when beginning a new
    // sequence, as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter* cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(scale_param);
      cont_h_param->set_name("h_conted_" + tm1s);
      cont_h_param->add_bottom("h_" + tm1s);
      cont_h_param->add_bottom("cont_" + ts);
      cont_h_param->add_top("h_conted_" + tm1s);
    }

    // Add layer to compute
    //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name("transform_" + ts);
      w_param->add_param()->set_name("W_hc");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("W_hc_h_" + tm1s);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter* input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_param);
      input_sum_layer->set_name("gate_input_" + ts);
      input_sum_layer->add_bottom("W_hc_h_" + tm1s);
      input_sum_layer->add_bottom("W_xc_x_" + ts);
      if (this->static_input_) {
        input_sum_layer->add_bottom("W_xc_x_static");
      }
      input_sum_layer->add_top("gate_input_" + ts);
    }

    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]
    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    {
      LayerParameter* lstm_unit_param = net_param->add_layer();
      lstm_unit_param->set_type("LSTMUnit");
      lstm_unit_param->add_bottom("c_" + tm1s);
      lstm_unit_param->add_bottom("gate_input_" + ts);
      lstm_unit_param->add_bottom("cont_" + ts);
      lstm_unit_param->add_top("c_" + ts);
      lstm_unit_param->add_top("h_" + ts);
      lstm_unit_param->set_name("unit_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
  }  // for (int t = 1; t <= this->T_; ++t)

  {
    LayerParameter* c_T_copy_param = net_param->add_layer();
    c_T_copy_param->CopyFrom(split_param);
    c_T_copy_param->add_bottom("c_" + format_int(this->T_));
    c_T_copy_param->add_top("c_T");
  }
  net_param->add_layer()->CopyFrom(output_concat_layer);
}

//geyijun@2017-05-11
//统计展开层的数据范围
template <typename Dtype>
void LSTMLayer<Dtype>::RangeInUnrolledNet() 
{
	vector<string> layer_names =  RecurrentLayer<Dtype>::unrolled_net_->layer_names();
	RecurrentLayer<Dtype>::unrolled_net_->RangeInLayers(layer_names,&max_params_,&max_data_,&min_data_);
}

//geyijun@2017-05-11
//统计展开层的数据进行定标
template <typename Dtype>
void LSTMLayer<Dtype>::CalcFlSign(int data_bw,int param_bw)
{
	printf("-----------------------------------------LSTMLayer CalcFlSign Enter-------------------------------------------------------------\n");
	vector<string> layer_names =  RecurrentLayer<Dtype>::unrolled_net_->layer_names();
	for(int i=0;i<layer_names.size();i++)
	{
		printf("[geyijun] layer[%d][%s] --->max_params[%f] max_data[%f] min_data[%f]\n",i,layer_names[i].c_str(),max_params_[i],max_data_[i],min_data_[i]);
	}
	printf("-----------------------------------------------------------------------------------------------------\n");

	//计算参数部分的Q
	map<int,vector<int> >::iterator iter = calc_params_fl_.find(param_bw);
	if (iter == calc_params_fl_.end())
	{
		printf("[geyijun] LSTMLayer::CalcFlSign--------->params's bw fl  \n");
		vector<int> params_fl;
		params_fl.resize(max_params_.size(),0);
		for (int layer_id = 0; layer_id < max_params_.size(); layer_id++)
		{
			if(max_params_[layer_id] != 0)
			{
				int il  = (int)ceil(log2(max_params_[layer_id])+1);
				int fl = param_bw-il;
				params_fl[layer_id] = fl;
				printf("[geyijun] LSTMLayer::layer[%d][%s] --->params_bw[%d] params_fl[%d]\n",
					layer_id,layer_names[layer_id].c_str(),param_bw,params_fl[layer_id]);
			}
		}
		calc_params_fl_.insert(map<int,vector<int> >::value_type(param_bw,params_fl));
	}
	printf("------------------------------------------------------------------------------------------------------\n");

	//计算数据部分的Q
	iter = calc_valid_data_fl_.find(data_bw);
	if (iter == calc_valid_data_fl_.end())
	{
		printf("[geyijun] LSTMLayer::CalcFlSign--------->data's bw fl  \n");	
		vector<int> valid_data_sign;
		vector<int> valid_data_fl;
		valid_data_sign.resize(max_params_.size(),0);
		valid_data_fl.resize(max_params_.size(),0);
		for (int layer_id = 0; layer_id < max_params_.size(); layer_id++)
		{
		
			int is_sign = (min_data_[layer_id]>=0)?0:1;
			valid_data_sign[layer_id] = is_sign;	
			if(max_data_[layer_id] == 0)
				max_data_[layer_id] = 1;
			int il  = (int)ceil(log2(max_data_[layer_id])+is_sign);	
			valid_data_fl[layer_id] = data_bw-il;	
			printf("[geyijun] LSTMLayer::layer[%d][%s] --->data_bw[%d] data_fl[%d] data_sign[%d]\n",
				layer_id,layer_names[layer_id].c_str(),data_bw,valid_data_fl[layer_id],valid_data_sign[layer_id]);
		}
		calc_valid_data_fl_.insert(map<int,vector<int> >::value_type(data_bw,valid_data_fl));		
		calc_valid_data_sign_.insert(map<int,vector<int> >::value_type(data_bw,valid_data_sign));	
	}
	printf("--------------------------------------------------LSTMLayer CalcFlSign Exit----------------------------------------------------\n");
}

template <typename Dtype>
void LSTMLayer<Dtype>::WriteFlSign(int data_bw,int param_bw,string filename)
{
	printf("--------------------------------------------------LSTMLayer WriteFlSign Enter----------------------------------------------------\n");

	//Write the net to a NetParameter
	NetParameter net_param;
	RecurrentLayer<Dtype>::unrolled_net_->ToProtoNotBlobs(&net_param);
	vector<string> layer_names =  RecurrentLayer<Dtype>::unrolled_net_->layer_names();

	//下面为了参数能够对上
	map<int,vector<int> >::iterator iter = calc_params_fl_.find(param_bw);
	assert(iter != calc_params_fl_.end());
	
	vector<int> v_params_bw;
	v_params_bw.resize(layer_names.size(),param_bw);
	vector<int> v_params_fl = iter->second;
	assert(v_params_fl.size() == layer_names.size());

	iter = calc_valid_data_fl_.find(data_bw);
	assert(iter != calc_valid_data_fl_.end());
	vector<int> v_data_bw;
	v_data_bw.resize(layer_names.size(),data_bw);
	vector<int> v_data_fl = iter->second;
	assert(v_data_fl.size() == layer_names.size());

	iter = calc_valid_data_sign_.begin();
	assert(iter != calc_valid_data_sign_.end());
	vector<int> v_data_sign = iter->second;
	assert(v_data_sign.size() == layer_names.size());
	Quantization::EditNetQuantizationParameter(&net_param,layer_names,
										v_params_bw,v_params_fl,
										v_data_bw,v_data_fl,v_data_sign);
	WriteProtoToTextFile(net_param, filename);
	printf("--------------------------------------------------LSTMLayer WriteFlSign Exit----------------------------------------------------\n");

	return ;
}

INSTANTIATE_CLASS(LSTMLayer);
REGISTER_LAYER_CLASS(LSTM);

}  // namespace caffe

