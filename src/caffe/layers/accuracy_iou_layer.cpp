#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_iou_layer.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyIOULayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	LOG(INFO) << "[geyijun] AccuracyIOULayer::LayerSetUp -------------------->1";
	iou_threshold_ = this->layer_param_.accuracy_iou_param().iou_threshold();
	iou_succ_ = 0;
	iou_total_ = 0;
	LOG(INFO) << "iou_threshold_="<<iou_threshold_;	
	vector<int> top_shape(0);  // AccuracyIOU is a scalar; 0 axes. 结果是标量(无轴)
	top[0]->Reshape(top_shape);
	LOG(INFO) << "[geyijun] AccuracyIOULayer::LayerSetUp -------------------->end";	
}

template <typename Dtype>
void AccuracyIOULayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	;
}

template <typename Dtype>
void AccuracyIOULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
	CHECK_EQ(bottom.size(),2)<< "bottom.size() should be 2";
	CHECK_EQ(bottom[0]->shape(1),4) << "bottom[0].shape(1) should be 4";
	CHECK_EQ(bottom[1]->shape(1)%4,0) << "bottom[0].shape(1) should be multiple of 4";
	
	const Dtype* bottom_region_lable = bottom[0]->cpu_data();
	const Dtype* bottom_region_select = bottom[1]->cpu_data();
	int batch_size = bottom[0]->shape(0);
	int max_region_num = bottom[1]->shape(1)/4;
	printf("[geyijun] : AccuracyIOULayer::Forward_cpu ---> batch_size=[%d ] max_region_num=[%d ]\n",batch_size,max_region_num);
	for (int item_id = 0; item_id < batch_size; ++item_id)
	{
		int region_lable_offset = bottom[0]->offset(item_id);
		int region_select_offset  = bottom[1]->offset(item_id);
		printf("[geyijun] : AccuracyIOULayer:: 	region_label_rect=[%d,%d,%d,%d] \n",\
							(int)(bottom_region_lable[region_lable_offset + 0]),
							(int)(bottom_region_lable[region_lable_offset + 1]),
							(int)(bottom_region_lable[region_lable_offset + 2]),
							(int)(bottom_region_lable[region_lable_offset + 3]));
		//遍历每一个选举出来的候选区域
		int max_iou = 0;
		for (int i = 0; i < max_region_num; ++i)
		{
			//计算两个矩形的IOU
			if(fabs(bottom_region_select[region_select_offset + i*4  + 0]) <= 0.0000001)
			{
				break;
			}
			printf("[geyijun] : AccuracyIOULayer:: 	Try: region_select_rect=[%d,%d,%d,%d] \n",\
								(int)(bottom_region_select[region_select_offset + i*4  + 0]),
								(int)(bottom_region_select[region_select_offset + i*4  + 1]),
								(int)(bottom_region_select[region_select_offset + i*4  + 2]),
								(int)(bottom_region_select[region_select_offset + i*4  + 3]));

			IOU_RECT_S intersection_rect;
			IOU_RECT_S union_rect;
			int ret = GetIntersectionUnion((int)(bottom_region_lable[region_lable_offset + 0]),
									(int)(bottom_region_lable[region_lable_offset + 1]),
									(int)(bottom_region_lable[region_lable_offset + 2]),
									(int)(bottom_region_lable[region_lable_offset + 3]),
									(int)(bottom_region_select[region_select_offset + i*4  + 0]),
									(int)(bottom_region_select[region_select_offset + i*4  + 1]),
									(int)(bottom_region_select[region_select_offset + i*4  + 2]),
									(int)(bottom_region_select[region_select_offset + i*4  + 3]),
									intersection_rect,union_rect);
			if (ret==0)
			{
				int iou = 100*(intersection_rect.right - intersection_rect.left)*(intersection_rect.bottom - intersection_rect.top) 
							/((union_rect.right - union_rect.left)*(union_rect.bottom - union_rect.top));
				if(iou>max_iou)
				{
					max_iou = iou;
				}
				printf("[geyijun] : AccuracyIOULayer:: 		Get: intersection_rect=[%d,%d,%d,%d] union_rect=[%d,%d,%d,%d]\n",
						intersection_rect.left,intersection_rect.top,intersection_rect.right,intersection_rect.bottom,
						union_rect.left,union_rect.top,union_rect.right,union_rect.bottom);
				printf("[geyijun] : AccuracyIOULayer:: 		iou=[%d]\n",iou,max_iou);
			}
			else
			{
				printf("[geyijun] : AccuracyIOULayer:: 		no iou\n");
			}
		}
		if(max_iou >= iou_threshold_)
		{
			iou_succ_++;
		}
		iou_total_++;
		printf("[geyijun] : AccuracyIOULayer:: 	max_iou=[%d],iou_succ_=[%d],iou_total_=[%d]\n",max_iou,iou_succ_,iou_total_);
	}
	//输出结果
	LOG(INFO) << "AccuracyIOU: " << (100.0*iou_succ_/iou_total_);
	top[0]->mutable_cpu_data()[0] = (100.0*iou_succ_/iou_total_);
}

//获取矩形的交集和并集
template <typename Dtype>
int AccuracyIOULayer<Dtype>::GetIntersectionUnion(
							int rectA_l,int rectA_t,int rectA_r,int rectA_b,
							int rectB_l,int rectB_t,int rectB_r,int rectB_b,
							IOU_RECT_S & intersection_rect,
							IOU_RECT_S & union_rect)
{
	//在x轴投影，得两个线段，再求两段的左端最大点和右端最小点    
	int x_left_max = rectB_l;
	if (rectA_l>rectB_l)
		x_left_max = rectA_l;
	int x_right_min = rectB_r;
	if (rectA_r<rectB_r)
		x_right_min = rectA_r;
	int y_top_max = rectB_t;
	if (rectA_t>rectB_t)
		y_top_max = rectA_t;
	int y_bottom_min = rectB_b;
	if (rectA_b<rectB_b)
		y_bottom_min = rectA_b;
	//判断相交    
	if ((x_left_max < x_right_min) && (y_top_max < y_bottom_min))
	{
		//交集        
		intersection_rect.left = x_left_max;
		intersection_rect.top = y_top_max;
		intersection_rect.right = x_right_min;
		intersection_rect.bottom = y_bottom_min;
		//并集(和交集相反)        
		union_rect.left = rectB_l;
		if (rectA_l<rectB_l)
			union_rect.left = rectA_l;
		union_rect.right = rectB_r;
		if (rectA_r>rectB_r)
			union_rect.right = rectA_r;
		union_rect.top = rectB_t;
		if (rectA_t<rectB_t)
			union_rect.top = rectA_t;
		union_rect.bottom = rectB_b;
		if (rectA_b>rectB_b)
			union_rect.bottom = rectA_b;
		return 0;
	}
	return -1;
}

INSTANTIATE_CLASS(AccuracyIOULayer);
REGISTER_LAYER_CLASS(AccuracyIOU);

}  // namespace caffe
