#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/region_select_layer.hpp"
/*
int debug_file_onoff(char *filename,float *score,int height,int width)
{
	printf("debug_file_onoff --->[%s] \n",filename);
	FILE *fp = NULL;
	if(fp=fopen(filename,"w+"))
	{
		for(int i=0;i<height;i++)
		{
			for(int j=0;j<width;j++)
			{
				fprintf(fp,"%c",(score[i*width + j]>0.6)?'x':'.');
			}
			fprintf(fp,"\n");
		}
	}
	else
	{
		printf("打开文件成败\n");
	}
	return 0;
}
*/
namespace caffe {

template <typename Dtype>
void RegionSelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	LOG(INFO) << "[geyijun] RegionSelectLayer::LayerSetUp -------------------->1";
	//获取参数
	area_min_ = this->layer_param_.regionselect_param().area_min();
	area_max_ = this->layer_param_.regionselect_param().area_max();
	max_count_ = this->layer_param_.regionselect_param().max_count();
	region_buff_ = (MY_REGION_INFO_S *)malloc(sizeof(MY_REGION_INFO_S)*max_count_);
	LOG(INFO) << "area_min_="<<area_min_ << " area_max_="<<area_max_ << " max_count_="<<max_count_;
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape(2);
	top_shape[0] = bottom_shape[0];		//batch_size
	top_shape[1] = 4*max_count_;		//lt 和rb 的坐标值
	top[0]->Reshape(top_shape);
	printf("[geyijun] : top_shape=[%d : %d ] \n",top_shape[0],top_shape[1]);
	LOG(INFO) << "[geyijun] RegionSelectLayer::LayerSetUp -------------------->end";
}

template <typename Dtype>
void RegionSelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) 
{
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape(2);
	top_shape[0] = bottom_shape[0];	//batch_size
	top_shape[1] = 4*max_count_;	//lt 和rb 的坐标值
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RegionSelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	vector<int> bottom_shape = bottom[0]->shape();
	CHECK_EQ(bottom[0]->shape(1),2) << "bottom.shape [1] should be 2";
	CHECK_EQ(bottom[0]->shape(0),top[0]->shape(0)) << "bottom.shape [0] should eq top.shape [0]";

	//先将输出清理
    	caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
	
	int batch_size = bottom_shape[0];
	for (int item_id = 0; item_id < batch_size; ++item_id) 
	{
		printf("[geyijun] : RegionSelectLayer::Forward_cpu ---> bottom_shape=[%d %d %d %d] sizeof(Dtype)=%d\n",
							bottom_shape[0],bottom_shape[1],bottom_shape[2],bottom_shape[3],
							(int)sizeof(Dtype));
		const Dtype* bottom_data = bottom[0]->cpu_data();
		int bottom_offset = bottom[0]->offset(item_id,1);	//第一通道是前景
		assert(sizeof(Dtype) == 4);
		
		//区域选举
		float *map = (float *)(bottom_data+bottom_offset);
		/*
		static  int write_debug = 0;
		if(write_debug  == 0)
		{
			write_debug  = 1;
			debug_file_onoff("score_map.txt",map,bottom_shape[2],bottom_shape[3]);
		}
		*/
		memset(region_buff_,0,sizeof(MY_REGION_INFO_S)*max_count_);
		int ret = RegionSelect(map, bottom_shape[3], bottom_shape[2], region_buff_,max_count_,60,area_min_,area_max_);
		if(ret < 0)
		{
			printf("[geyijun] : RegionSelectLayer::Forward_cpu --->RegionSelect failed!!!\n");
		}
		printf("[geyijun] : RegionSelectLayer::Forward_cpu ---> RegionSelect=[%d ] \n",ret);
		//输出结果
		Dtype* top_data = top[0]->mutable_cpu_data();
		int top_offset = top[0]->offset(item_id);
		for(int i=0;i<ret;i++)
		{
			printf("[geyijun] : RegionSelectLayer::			Rect[%d]====[%d,%d,%d,%d]=== ! \n",i,
													region_buff_[i].rect.left,
													region_buff_[i].rect.top,
													region_buff_[i].rect.right,
													region_buff_[i].rect.bottom);
			*(top_data + top_offset + i*4 + 0) = region_buff_[i].rect.left;
			*(top_data + top_offset + i*4 + 1) = region_buff_[i].rect.top;
			*(top_data + top_offset + i*4 + 2) = region_buff_[i].rect.right;
			*(top_data + top_offset + i*4 + 3) = region_buff_[i].rect.bottom;		
		}
	}
}
INSTANTIATE_CLASS(RegionSelectLayer);
REGISTER_LAYER_CLASS(RegionSelect);

}  // namespace caffe
