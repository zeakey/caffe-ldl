#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/neural_decision_forest_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe
{
	__device__ int sub2ind_(int n, int c, int h, int w, int N, int C, int H, int W) 
	{
  		return  ((n * C + c) * H + h) * W + w;
	}

	__device__ int ind2sub_(int index, int C, int H, int W, int* n, int* c, int* h, int* w) 
	{
	  *w = index % W;
	  *h = (index / W) % H;
	  *c = (index / (W*H)) % C;
	  *n = index / (C*W*H);
	  return 0;
	}

	template <typename Dtype>
	__global__ void kernel_routing_(const int num, const int trees, const int dn_channel_dim,
		const int spatial_h_dim, const int spatial_w_dim, const int leaf_nodes_per_tree, 
		const int split_nodes_pre_tree, const Dtype* dn_data, const Dtype* sub_dim_data, Dtype* routing_split_out, Dtype* routing_leaf_out) 
	{
		int spatial_dim = spatial_h_dim * spatial_w_dim;
		CUDA_KERNEL_LOOP(index, num * spatial_dim * trees) 
		{
			int n, s, t, j;
			int idx = index;
			
			ind2sub_(idx, spatial_dim, trees, 1, &n, &s, &t, &j);
			
			for (int current_offset = 0; current_offset < split_nodes_pre_tree; current_offset++)
			{
				int left_child_offset = 2 * current_offset + 1;
				int right_child_offset = 2 * current_offset + 2;
				
				int sub_dim_offset = (int) sub_dim_data[sub2ind_(t, current_offset, 0, 0, trees, split_nodes_pre_tree, 1, 1)];

				Dtype dn = dn_data[sub2ind_(n, sub_dim_offset, s, 0, num, dn_channel_dim, spatial_dim, 1)];
				if (right_child_offset < split_nodes_pre_tree)
				{
					routing_split_out[sub2ind_(n, s, t, left_child_offset, num, spatial_dim, trees, split_nodes_pre_tree)] 
					= routing_split_out[sub2ind_(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * dn;

					routing_split_out[sub2ind_(n, s, t, right_child_offset, num, spatial_dim, trees, split_nodes_pre_tree)] = 
					routing_split_out[sub2ind_(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * ((Dtype) 1.0 - dn);
				}
				else
				{
					right_child_offset -= split_nodes_pre_tree;
					left_child_offset -= split_nodes_pre_tree;
					routing_leaf_out[sub2ind_(n, s, t, left_child_offset, num, spatial_dim, trees, leaf_nodes_per_tree)] 
					= routing_split_out[sub2ind_(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * dn;
					routing_leaf_out[sub2ind_(n, s, t, right_child_offset, num, spatial_dim, trees, leaf_nodes_per_tree)] 
					= routing_split_out[sub2ind_(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * ((Dtype) 1.0 - dn);
				}
			}
		}
	}

	template <typename Dtype>
	__global__ void kernel_transform(const int num, const int channel,
		const int height, const int width, Dtype* prediction_in, Dtype* prediction_out)
	{
		CUDA_KERNEL_LOOP(index, num * channel * height * width)
		{
			int n = index / (channel * height * width);
			int c = (index / (height * width)) % channel;
			int s = index % (height * width);
			prediction_out[index] = prediction_in[n * height * width * channel + s * channel + c];
		}
	}

	template <typename Dtype>
	void NeuralDecisionForestLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
        #if 1
		Blob<Dtype> * output_prob_ = top[0];
		Dtype* output_prob_data = output_prob_->mutable_gpu_data();
		

		sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
		const Dtype* dn_data = dn_->gpu_data();
		Dtype* routing_split_prob_data = routing_split_prob_.mutable_gpu_data();
		Dtype* routing_leaf_prob_data = routing_leaf_prob_.mutable_gpu_data();

		const Dtype* class_label_distr_data = class_label_distr_->gpu_data();
		const Dtype* sub_dimensions_data = sub_dimensions_->gpu_data();
		
		
		Dtype* forest_prediction_prob_data = forest_prediction_prob_.mutable_gpu_data();

		kernel_routing_<Dtype> << <CAFFE_GET_BLOCKS(num_outer_ * num_inner_ * num_trees_),
			CAFFE_CUDA_NUM_THREADS >> >(num_outer_, num_trees_, num_dims_, bottom[0]->height(), bottom[0]->width(), num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_data, sub_dimensions_data,
			routing_split_prob_data, routing_leaf_prob_data);
		
	
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_outer_ * num_inner_, num_classes_, num_trees_ * num_leaf_nodes_per_tree_,
			(Dtype)1.0, routing_leaf_prob_data, class_label_distr_data, (Dtype)0.0, forest_prediction_prob_data);

		caffe_gpu_scal(num_outer_ * num_inner_ * num_classes_, (Dtype)1.0 / num_trees_, forest_prediction_prob_data);
		CHECK_EQ(forest_prediction_prob_.count(), output_prob_->count());

		kernel_transform<Dtype> << < CAFFE_GET_BLOCKS(num_outer_ * num_inner_ * num_classes_),
			CAFFE_CUDA_NUM_THREADS >> > (num_outer_, num_classes_, bottom[0]->height(), bottom[0]->width(),
			forest_prediction_prob_data, output_prob_data);
#else
		Forward_cpu(bottom, top);
#endif
	}

	template <typename Dtype>
	void NeuralDecisionForestLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Backward_cpu(top, propagate_down, bottom);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NeuralDecisionForestLayer);
}