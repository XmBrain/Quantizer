#ifndef CAFFE_REGION_SELECT_LAYER_HPP_
#define CAFFE_REGION_SELECT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"

#include "region_select.h"

namespace caffe {

//geyijun@2016-12-14
//该层根据的SoftMax的计算结果调用库lib_region_select
//选取出目标对象的矩形坐标!!!
template <typename Dtype>
class RegionSelectLayer : public Layer<Dtype> {
public:
	explicit RegionSelectLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
							const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
							const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "RegionSelect"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    			NOT_IMPLEMENTED;}
protected:	
	int area_min_;
	int area_max_;
	int max_count_;
	MY_REGION_INFO_S *region_buff_;
};

}  // namespace caffe

#endif  // CAFFE_COMMON_LAYERS_HPP_
