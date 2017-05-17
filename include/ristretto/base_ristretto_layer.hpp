#ifndef CAFFE_BASE_RISTRETTO_LAYER_HPP_
#define CAFFE_BASE_RISTRETTO_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/lstm_layer.hpp"
#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

//geyijun@2016-12-13
template <typename Dtype>
class BaseRistrettoLayer
{
public:
	explicit BaseRistrettoLayer();
protected:
	void QuantizeLayerOutputs_cpu(Dtype* data, const int count);
	void QuantizeWeights_cpu(vector<shared_ptr<Blob<Dtype> > > weights_quantized,const bool bias_term = true);
	void Trim2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width,int fl,int is_sign,int needround);

	// 量化参数
	int	bw_params_;
	int	fl_params_;
	int	bw_layer_out_;
	int	fl_layer_out_;
	int	is_sign_out;

	//geyijun@2017-04-01
	//对bias 也要定标
	int	init_flag_bias_;
	int	bw_params_bias_;
	int 	fl_params_bias_;
};

/**
 * @brief Convolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class ConvolutionRistrettoLayer : public ConvolutionLayer<Dtype>,
									public BaseRistrettoLayer<Dtype> 
{
public:
	explicit ConvolutionRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "ConvolutionRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
};

/**
 * @brief Deconvolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class DeconvolutionRistrettoLayer : public DeconvolutionLayer<Dtype>,public BaseRistrettoLayer<Dtype> 
{
public:
	explicit DeconvolutionRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "DeconvolutionRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};

/**
 * @brief Inner product (fully connected) layer with quantized layer parameters
 * and activations.
 */
template <typename Dtype>
class InnerProductRistrettoLayer : public InnerProductLayer<Dtype>,
							public BaseRistrettoLayer<Dtype>
{
public:
	explicit InnerProductRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "InnerProductRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};

template <typename Dtype>
class LRNRistrettoLayer : public LRNLayer<Dtype>,
     							 public BaseRistrettoLayer<Dtype>
{
public:
	explicit LRNRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "LRNRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};

template <typename Dtype>
class PoolingRistrettoLayer : public PoolingLayer<Dtype>,
     							 public BaseRistrettoLayer<Dtype>
{
public:
	explicit PoolingRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "PoolingRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};

template <typename Dtype>
class ReLURistrettoLayer : public ReLULayer<Dtype>,
     							 public BaseRistrettoLayer<Dtype>
{
public:
	explicit ReLURistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "ReLURistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};

template <typename Dtype>
class DataRistrettoLayer : public DataLayer<Dtype>,
									public BaseRistrettoLayer<Dtype> 
{
public:
	explicit DataRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "DataRistretto"; }
protected:
  	virtual void load_batch(Batch<Dtype>* batch);
};

template <typename Dtype>
class SoftmaxRistrettoLayer : public SoftmaxLayer<Dtype>,
									public BaseRistrettoLayer<Dtype> 
{
public:
	explicit SoftmaxRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "SoftmaxRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};

template <typename Dtype>
class ConcatRistrettoLayer : public ConcatLayer<Dtype>,
									public BaseRistrettoLayer<Dtype> 
{
public:
	explicit ConcatRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "ConcatRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};


template <typename Dtype>
class LSTMRistrettoLayer : public LSTMLayer<Dtype>,
									public BaseRistrettoLayer<Dtype> 
{
public:
	explicit LSTMRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "LSTMRistretto"; }
protected:
	virtual void FillUnrolledNet(NetParameter* net_param) const;
	
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};

template <typename Dtype>
class LSTMUnitRistrettoLayer : public LSTMUnitLayer<Dtype>,
									public BaseRistrettoLayer<Dtype> 
{
public:
	explicit LSTMUnitRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "LSTMUnitRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
};

template <typename Dtype>
class SliceRistrettoLayer : public SliceLayer<Dtype>,
									public BaseRistrettoLayer<Dtype> 
{
public:
	explicit SliceRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "SliceRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
};

template <typename Dtype>
class EltwiseRistrettoLayer : public EltwiseLayer<Dtype>,
									public BaseRistrettoLayer<Dtype> 
{
public:
	explicit EltwiseRistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "EltwiseRistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
};

template <typename Dtype>
class PReLURistrettoLayer : public PReLULayer<Dtype>,
     							 public BaseRistrettoLayer<Dtype>
{
public:
	explicit PReLURistrettoLayer(const LayerParameter& param);
	virtual inline const char* type() const { return "PReLURistretto"; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);	
};

}  // namespace caffe

#endif  // CAFFE_BASE_RISTRETTO_LAYER_HPP_
