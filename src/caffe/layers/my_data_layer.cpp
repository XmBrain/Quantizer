//geyijun@2016-07-21
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>

#include <assert.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>

#include "json/value.h"
#include "json/reader.h"
#include "json/writer.h"

#include "caffe/layers/data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

//叉乘法判断两个点是否在一个线段AB的同侧
bool isInSameSide(Point segmentA,Point segmentB,Point point1,Point point2)
{
	//printf(" AB ===> (%d,%d) --->(%d,%d) \n",segmentA.m_x,segmentA.m_y,segmentB.m_x,segmentB.m_y);
	//printf(" point1 ===> (%d,%d) \n",point1.m_x,point1.m_y);
	//printf(" point2 ===> (%d,%d) \n",point2.m_x,point2.m_y);
	int a = (point1.m_x - segmentA.m_x)*(point1.m_y - segmentB.m_y) - (point1.m_y - segmentA.m_y)*(point1.m_x - segmentB.m_x); 
	int b = (point2.m_x - segmentA.m_x)*(point2.m_y - segmentB.m_y) - (point2.m_y - segmentA.m_y)*(point2.m_x - segmentB.m_x);
	//printf("a: %d === a:%d \n",a,b);
	if (((a > 0)&&(b > 0))
	||((a < 0)&&(b < 0)))
		return true;
	else
		return false;
}

//判断点是否在外包四边形内  
bool isPointInArea(Point point,FourPointArea &area)
{
	for (int index=0;index<4;index++)
	{
		if(false == isInSameSide(area.m_pset[index],area.m_pset[(index+1)%4],point,area.m_pset[(index+2)%4]))           
			return false;
	}
	return true;
}

template <typename Dtype>
MyDataLayer<Dtype>::~MyDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void MyDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) 
{
	LOG(INFO) << "[geyijun] MyDataLayer::DataLayerSetUp -------------------->1";
	const int batch_size = this->layer_param_.my_data_param().batch_size();
	const bool is_color  = this->layer_param_.my_data_param().is_color();
	const string root_folder = this->layer_param_.my_data_param().root_folder();
	const int lable_mode = this->layer_param_.my_data_param().lable_mode();

	plate_area_map_.clear();
	//一行里面包含有--->(图片文件名【空格】配置文件名)
	const string& source = this->layer_param_.my_data_param().source();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	string image_filename;
	string cfg_filename;
	while (infile >> image_filename >> cfg_filename) 
	{
		lines_.push_back(std::make_pair(image_filename, cfg_filename));

		//事先备好map表这样快一点
		FourPointArea plate_area;
		get_plate_area(root_folder+cfg_filename,plate_area);
		plate_area_map_[cfg_filename] = plate_area;
	}

	if (this->layer_param_.my_data_param().shuffle()) 
	{
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		ShuffleImages();
	}
	LOG(INFO) << "A total of " << lines_.size() << " images.";

	lines_id_ = 0;
	// Check if we would need to randomly skip a few data points
	if (this->layer_param_.my_data_param().rand_skip()) 
	{
		unsigned int skip = caffe_rng_rand() %
		this->layer_param_.my_data_param().rand_skip();
		LOG(INFO) << "Skipping first " << skip << " data points.";
		CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
		lines_id_ = skip;
	}

	// Read an image, and use it to initialize the top blob.
	cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, is_color);
	CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

	// Use data_transformer to infer the expected blob shape from a cv_image.
	vector<int> image_shape = this->data_transformer_->InferBlobShape(cv_img);
	this->transformed_data_.Reshape(image_shape);
	printf("[geyijun] : image_shape  =[%d : %d : %d : %d ] \n",image_shape[0],image_shape[1],image_shape[2],image_shape[3]);

	// Reshape prefetch_data and top[0] according to the batch_size.
	CHECK_GT(batch_size, 0) << "Positive batch size required";
	image_shape[0] = batch_size;		//修改
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	{
		this->prefetch_[i].data_.Reshape(image_shape);
	}
	top[0]->Reshape(image_shape);

	// label 
	//注意::: 这里标签的shape  [batch_size,1,H,W ]
	vector<int> label_shape;
	if(lable_mode == 0)//输出每一个像素点的分类
	{
		label_shape.resize(4);
		label_shape[0] = batch_size;
		label_shape[1] = 1;
		label_shape[2] = image_shape[2];
		label_shape[3] = image_shape[3];		
		printf("[geyijun] : label_map_shape =[%d : %d : %d : %d ] \n",label_shape[0],label_shape[1],label_shape[2],label_shape[3]);
	}
	else	//输出坐标
	{
		label_shape.resize(2);
		label_shape[0] = batch_size;
		label_shape[1] = 4;
		printf("[geyijun] : label_pos_shape =[%d : %d ] \n",label_shape[0],label_shape[1]);
	}
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	{
		this->prefetch_[i].label_.Reshape(label_shape);
	}
	top[1]->Reshape(label_shape);


	LOG(INFO) << "[geyijun] MyDataLayer::DataLayerSetUp -------------------->end";
}

template <typename Dtype>
void MyDataLayer<Dtype>::ShuffleImages() 
{
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MyDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) 
{
	//printf("[geyijun] load_batch-------------------->1\n");
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());
	const int batch_size  = this->layer_param_.my_data_param().batch_size();
	const bool is_color  = this->layer_param_.my_data_param().is_color();
	const string root_folder = this->layer_param_.my_data_param().root_folder();
	const int lable_mode = this->layer_param_.my_data_param().lable_mode();

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	Dtype* prefetch_label = batch->label_.mutable_cpu_data();	
	
	// datum scales
	const int lines_size = lines_.size();
	for (int item_id = 0; item_id < batch_size; ++item_id) 
	{
		//printf("[geyijun] load_batch-------------------->a\n");	
		//读取车牌的区域
		FourPointArea plate_area;
		map<string, FourPointArea>::iterator iter = plate_area_map_.find(lines_[lines_id_].second);
		if (iter == plate_area_map_.end()) 
		{
			LOG(ERROR) << "Could not get_plate_area: " << lines_[lines_id_].second;
		}
		else
		{
			 plate_area = iter->second;
		}
		//printf("[geyijun] load_batch-------------------->label[%s]\n",lines_[lines_id_].second.c_str());

		//读取数据文件
		CHECK_GT(lines_size, lines_id_);
		cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,is_color);
		CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
		vector<int> image_shape = this->data_transformer_->InferBlobShape(cv_img);
		vector<int> batch_data_shape = batch->data_.shape();
		CHECK_EQ(batch_data_shape[2], image_shape[2]) << "Image Shape Not match with BatchData 1";
		CHECK_EQ(batch_data_shape[3], image_shape[3]) << "Image Shape Not match with BatchData 2";
			
		//printf("[geyijun] load_batch-------------------->b.2\n");	
		//对数据文件进行转换处理
		int offset = batch->data_.offset(item_id);
		this->transformed_data_.set_cpu_data(prefetch_data + offset);			
		this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

		//读取标签文件构造标签数据
		int label_offset = batch->label_.offset(item_id);
		if(lable_mode == 0)	//输出每一个像素点的分类
		{
			build_label(plate_area,image_shape[3],image_shape[2],prefetch_label + label_offset); 
		}
		else					//输出坐标
		{
			Point lt,rb;
			plate_area.GetBoundRect(lt,rb);
			*(prefetch_label + label_offset + 0) = lt.m_x*image_shape[3]/8192;
			*(prefetch_label + label_offset + 1) = lt.m_y*image_shape[2]/8192;
			*(prefetch_label + label_offset + 2) = rb.m_x*image_shape[3]/8192;
			*(prefetch_label + label_offset + 3) = rb.m_y*image_shape[2]/8192;
		}
		//printf("[geyijun] load_batch-------------------->c\n");
		
		// go to the next iter
		lines_id_++;
		if (lines_id_ >= lines_size) 
		{
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.my_data_param().shuffle()) 
			{
				ShuffleImages();
			}
		}
		//printf("[geyijun] load_batch-------------------->d\n");		
	}
	//printf("[geyijun] load_batch-------------------->e\n");	
	//LOG(INFO) << "[geyijun] load_batch -------------------->end" ;
}

template <typename Dtype>
bool MyDataLayer<Dtype>::get_plate_area(const string& cfgfile,FourPointArea &area)
{
	FILE *file = fopen(cfgfile.c_str(), "rb" );
	CHECK(file) << "Could not open " << cfgfile;
   	fseek( file, 0, SEEK_END );
   	int size = ftell( file );

   	fseek( file, 0, SEEK_SET );
 	char pvData[8192] = {0,};
	int ret = fread( pvData, 1, size, file );
	fclose(file);
	if (ret  == size )
	{
		Json::Reader	reader;
		Json::Value 	jtable;
		bool bRet = reader.parse(pvData, jtable);
		if ((bRet == true) 
			&& (jtable.isMember("Plate"))
			&& (jtable["Plate"].isArray())
			&& (jtable["Plate"].size() == 8))	//这是8个数字的数组(每两个数组表示一个点)
		{
			area = FourPointArea(jtable["Plate"][(unsigned int)0].asInt(),
												jtable["Plate"][1].asInt(),
												jtable["Plate"][2].asInt(),
												jtable["Plate"][3].asInt(),
												jtable["Plate"][4].asInt(),
												jtable["Plate"][5].asInt(),
												jtable["Plate"][6].asInt(),
												jtable["Plate"][7].asInt());
			return true;		
		}
		else
		{
			CHECK(false) << "Invalid json body" << cfgfile;
			return false;		
		}
	}
	else
	{
		CHECK(false) << "Could not read " << cfgfile;
		return false;	
	}
	return false;
}

template <typename Dtype>
void MyDataLayer<Dtype>::build_label(FourPointArea plate_area,int width,int height,Dtype* label)
{
	
	/*
	printf("build_label ===> platearea: [%d:%d ] [%d:%d ] [%d:%d ] [%d:%d ]\n",plate_area.m_pset[0].m_x,plate_area.m_pset[0].m_y \
									,plate_area.m_pset[1].m_x,plate_area.m_pset[1].m_y \
									,plate_area.m_pset[2].m_x,plate_area.m_pset[2].m_y \
									,plate_area.m_pset[3].m_x,plate_area.m_pset[3].m_y);
	printf("build_label ===> width[%d] height[%d]\n",width,height);
	*/
	caffe_set(width*height,Dtype(0),label);

	//坐标转成实际图像的大小(物理坐标)
	plate_area.m_pset[0].m_x = plate_area.m_pset[0].m_x*width/8192;
	plate_area.m_pset[0].m_y = plate_area.m_pset[0].m_y*height/8192;
	plate_area.m_pset[1].m_x = plate_area.m_pset[1].m_x*width/8192;
	plate_area.m_pset[1].m_y = plate_area.m_pset[1].m_y*height/8192;
	plate_area.m_pset[2].m_x = plate_area.m_pset[2].m_x*width/8192;
	plate_area.m_pset[2].m_y = plate_area.m_pset[2].m_y*height/8192;
	plate_area.m_pset[3].m_x = plate_area.m_pset[3].m_x*width/8192;
	plate_area.m_pset[3].m_y = plate_area.m_pset[3].m_y*height/8192;
	Point lt,rb;
	plate_area.GetBoundRect(lt,rb);
	for (int y=lt.m_y;y<rb.m_y;y+=1)
	{
		for (int x=lt.m_x;x<rb.m_x;x+=1)
		{
			if (isPointInArea(Point(x,y),plate_area))
			{
				label[y * width + x]  = 1;
				//printf("x");
			}
			else
			{
				;//printf(".");
			}
		}
		//printf("\n"); 
	}
	/*static int a = 0;
	if(a == 0)
	{
		a = 1;
		FILE *file = fopen("1.txt", "wb" );
		for (int y=0;y<height;y+=1)
		{
			for (int x=0;x<width;x+=1)
			{
				if (isPointInArea(Point(x,y),plate_area))
				{
					label[y * width + x]  = 1;
					fprintf(file,"x");
				}
				else
				{
					fprintf(file,".");
				}
			}
			fprintf(file,"\n");
		}
		fclose(file);
	}
	*/
    	//printf("[geyijun] build_label------------------>end\n"); 
	return ;
}

INSTANTIATE_CLASS(MyDataLayer);
REGISTER_LAYER_CLASS(MyData);

}  // namespace caffe

