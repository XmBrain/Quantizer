#ifndef CAFFE_ACCURACY_IOU_LAYER_HPP_
#define CAFFE_ACCURACY_IOU_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

typedef struct IOU_RECT
{
	unsigned short left;	
	unsigned short top;	
	unsigned short right;
	unsigned short bottom;
}IOU_RECT_S;

template <typename Dtype>
class AccuracyIOULayer : public Layer<Dtype> {
public:
	explicit AccuracyIOULayer(const LayerParameter& param): Layer<Dtype>(param) {}
  	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      							const vector<Blob<Dtype>*>& top);
  	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      							const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "AccuracyIOU"; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 1; }
protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      								const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      	{NOT_IMPLEMENTED; }
private:	
	int GetIntersectionUnion(int rectA_l,int rectA_t,int rectA_r,int rectA_b,
								int rectB_l,int rectB_t,int rectB_r,int rectB_b,
								IOU_RECT_S & intersection_rect,
								IOU_RECT_S & union_rect);	
protected:
	int iou_threshold_;
	int iou_succ_;	//
	int iou_total_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_IOU_LAYER_HPP_