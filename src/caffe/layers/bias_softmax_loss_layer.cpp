#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BiasSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
//geyijun@2016-07-12
//这里要创建的是Softmax 实例，就像SoftmaxWithLoss一样
//在SoftmaxWithLoss 内存在一个Softmax的实例(封装的一层)
//所以这里其实也是一样的，是对Softmax的封装
//而不是BiasSoftmax.
//softmax_param.set_type("BiasSoftmax");  
  softmax_param.set_type("Softmax"); 	
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();

  //geyijun@2016-07-25
  //负标签的编号(当前场景中0-背景; 1-前景)
  //customized parameters:
  neg_id = 0;
}

template <typename Dtype>
void BiasSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void BiasSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0, count_pos = 0, count_neg = 0;
  Dtype loss = 0;
  //pos/neg loss for single image
  Dtype temp_loss_neg = 0; 
  Dtype temp_loss_pos = 0;
  //pos/neg loss for a batch
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  //printf("[geyijun@2016-07-12]----------------------------->Forward_cpu outer_num_[%d] inner_num_ [%d]\n",outer_num_,inner_num_);    

  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      if ( label_value == neg_id) {
        temp_loss_neg -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
        count_neg++;
      } else {
        temp_loss_pos -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
        count_pos++;
      }
    }
    loss_neg += temp_loss_neg;
    loss_pos += temp_loss_pos;
    temp_loss_neg = 0;
    temp_loss_pos = 0;
    //printf("[geyijun@2016-07-12]--->[%d]:: loss_neg[%f] loss_pos[%f] \n",i,temp_loss_neg,temp_loss_pos);    
  }
  count = count_pos + count_neg;
  loss = loss_pos + loss_neg;
  //printf("[geyijun@2016-07-12]--->[total]::loss[%f]  loss_neg[%f] loss_pos[%f] \n",loss, loss_neg,loss_pos);

  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
    //printf("[geyijun@2016-07-12]----------------------------->1 avgloss[%f] loss[%f] count[%d] \n",loss / count,loss,count);    
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
    //printf("[geyijun@2016-07-12]----------------------------->2 avgloss[%f] loss[%f] count[%d] \n",loss / count,loss,count);    
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void BiasSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count_neg = 0, count_pos = 0, count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0;j < inner_num_; ++j) {
        if (label[i * inner_num_ + j] == neg_id)
          count_neg++;
        else
        {
        	//if(count_pos == 0)
		//	printf("[geyijun@2016-07-25]----------------------------->first pos j=[%d]\n",j); 		
        	count_pos++;
        }
      }
      //geyijun@2016-07-12	  
      Dtype beta[2] = { (Dtype)count_pos / (count_pos + count_neg),\
                        (Dtype)count_neg / (count_pos + count_neg) };
      //int beta[2] = { count_pos / (count_pos + count_neg),\
      //                count_neg / (count_pos + count_neg) };
      //printf("[geyijun@2016-07-25]----------------------------->outer_num_[%d] inner_num_[%d]\n",outer_num_,inner_num_); 
      //printf("[geyijun@2016-07-25]----------------------------->[%d] count_neg[%d] count_pos[%d]\n",count_neg+count_pos,count_neg,count_pos); 
      //printf("[geyijun@2016-07-25]----------------------------->beta[%f] [%f]\n",beta[0],beta[1]); 
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        }
	else {
          for (int ch = 0; ch < bottom[0]->shape(softmax_axis_); ++ch) {
            bottom_diff[i * dim + ch * inner_num_ + j] = 
                  - ( beta[label_value != neg_id] * (label_value == ch) \
                  - beta[label_value != neg_id] * bottom_diff[i * dim + ch * inner_num_ + j] );
          }
        }
      }
    }
    count = count_neg + count_pos;
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BiasSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(BiasSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(BiasSoftmaxWithLoss);

}  // namespace caffe
