/*
 * @author Wei Shen, Kai Zhao
 * LDLForest is open source code; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with LDLForest .  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
*/
#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/layers/neural_decision_distr_learning_forest_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe{
template <typename Dtype>
bool isdiff(Dtype x, Dtype y) {
  Dtype THRES = 0.0000001;
  return std::abs(x - y) >= THRES;
}

template <typename Dtype>
Dtype bdifference_dl(int count, const Dtype* X, const Dtype* Y) {
    Dtype* Z = new Dtype[count];
    caffe_sub(count, X, Y, Z);
    Dtype d = caffe_cpu_asum(count, Z) / count;
    delete [] Z;
    return d;
}
__device__ int sub2ind_dl(int n, int c, int h, int w, int N, int C, int H, int W) {
  return  ((n * C + c) * H + h) * W + w;
}

__device__ int ind2sub_dl(int index, int C, int H, int W, int* n, int* c, int* h, int* w) {
  *w = index % W;
  *h = (index / W) % H;
  *c = (index / (W*H)) % C;
  *n = index / (C*W*H);
  return 0;
}

template <typename Dtype>
__global__ void kernel_updata_all_dl(int num_outer_iter, int num_inner_iter,
          int num_trees, int num_leaf_nodes_per_tree, int num_class, const Dtype* routing_data, 
          const Dtype* class_label_distr_data, Dtype* pred_data) {
  int count = num_outer_iter * num_inner_iter * num_trees * num_class;
  CUDA_KERNEL_LOOP(index, count) {
    int t, k, i, j;
    int idx = index;
    ind2sub_dl(idx, num_inner_iter, num_trees, num_class, &i, &k, &t, &j);
    int pred_idx = sub2ind_dl(i, k, t, j, num_outer_iter, num_inner_iter, num_trees, num_class);
    for(int l = 0; l < num_leaf_nodes_per_tree; l++) {
        int routing_idx = sub2ind_dl(i, k, t, l, num_outer_iter, num_inner_iter, num_trees, num_leaf_nodes_per_tree);
        int distr_idx = sub2ind_dl(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1);
        pred_data[pred_idx] += routing_data[routing_idx] * class_label_distr_data[distr_idx];
    }
  }
}
template <typename Dtype> 
__global__ void kernel_update_leaf_dl(int num_trees, int num_leaf_nodes_per_tree, int num_class, int num_outer, int num_inner,
    const Dtype* class_label_distr_data, const Dtype* label_data, const Dtype* routing_leaf_prob_data, const Dtype* tree_prediction_prob_data, 
    Dtype* class_label_distr_temp_data) {
    CUDA_KERNEL_LOOP(index, num_trees * num_leaf_nodes_per_tree * num_class) {
        int t, l, j, i, k;
        int idx = index;
        ind2sub_dl(idx, num_trees, num_leaf_nodes_per_tree, num_class, &i, &t, &l, &j);
        for (i = 0; i < num_outer; i++) {
            for (k = 0; k < num_inner; k++) {
                class_label_distr_temp_data[sub2ind_dl(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1)] 
                += label_data[sub2ind_dl(i, k, j, 0, num_outer, num_inner, num_class, 1)] 
                * (class_label_distr_data[sub2ind_dl(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1)] 
                * routing_leaf_prob_data[sub2ind_dl(i, k, t, l, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree)] 
                / max(tree_prediction_prob_data[sub2ind_dl(i, k, t, j, num_outer, num_inner, num_trees, num_class)], Dtype(FLT_MIN)));
            }
        }
    }
}

template <typename Dtype>
__global__ void kernel_routing_dl(const int num, const int trees, const int dn_channel_dim,
    const int spatial_h_dim, const int spatial_w_dim, const int leaf_nodes_per_tree, 
    const int split_nodes_pre_tree, const Dtype* dn_data, const Dtype* sub_dim_data, Dtype* routing_split_out, Dtype* routing_leaf_out) {
    int spatial_dim = spatial_h_dim * spatial_w_dim;
    CUDA_KERNEL_LOOP(index, num * spatial_dim * trees) {
        int n, s, t, j;
        int idx = index;
        ind2sub_dl(idx, spatial_dim, trees, 1, &n, &s, &t, &j);
        for (int current_offset = 0; current_offset < split_nodes_pre_tree; current_offset++) {
            int left_child_offset = 2 * current_offset + 1;
            int right_child_offset = 2 * current_offset + 2;
            int sub_dim_offset = (int) sub_dim_data[sub2ind_dl(t, current_offset, 0, 0, trees, split_nodes_pre_tree, 1, 1)];
            Dtype dn = dn_data[sub2ind_dl(n, sub_dim_offset, s, 0, num, dn_channel_dim, spatial_dim, 1)];
            if (right_child_offset < split_nodes_pre_tree) {
                routing_split_out[sub2ind_dl(n, s, t, left_child_offset, num, spatial_dim, trees, split_nodes_pre_tree)] 
                = routing_split_out[sub2ind_dl(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * dn;
                routing_split_out[sub2ind_dl(n, s, t, right_child_offset, num, spatial_dim, trees, split_nodes_pre_tree)] = 
                routing_split_out[sub2ind_dl(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * ((Dtype) 1.0 - dn);
            } else {
                right_child_offset -= split_nodes_pre_tree;
                left_child_offset -= split_nodes_pre_tree;
                routing_leaf_out[sub2ind_dl(n, s, t, left_child_offset, num, spatial_dim, trees, leaf_nodes_per_tree)] 
                = routing_split_out[sub2ind_dl(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * dn;
                routing_leaf_out[sub2ind_dl(n, s, t, right_child_offset, num, spatial_dim, trees, leaf_nodes_per_tree)] 
                = routing_split_out[sub2ind_dl(n, s, t, current_offset, num, spatial_dim, trees, split_nodes_pre_tree)] * ((Dtype) 1.0 - dn);
            }
        }
    }
}

template <typename Dtype>
__global__ void kernel_bottom_diff_dl(int num_outer, int num_inner, int num_trees, int num_leaf_nodes_per_tree,
  int num_split_nodes_per_tree, int num_nodes_pre_tree, int num_class, int t, int N, int C, int H, int W,
  const Dtype* class_label_distr_data, const Dtype* routing_leaf_data,
  const Dtype* dn_data, const Dtype* sub_dim_data, const Dtype* tree_prediction_data, 
  const Dtype* label_data, Dtype* inter_var_data, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, num_outer * num_inner) {
        int idx = index;
        int i, k, j, l;
        ind2sub_dl(idx, 1, num_outer, num_inner, &l, &j, &i, &k);
        int pred_idx = sub2ind_dl(i, k, t, j, num_outer, num_inner, num_trees, num_class);
        for (l = 0; l < num_leaf_nodes_per_tree; ++l) {
            for (j = 0; j < num_class; ++j) {
                int inter_idx = sub2ind_dl(i, k, t * num_nodes_pre_tree + num_split_nodes_per_tree + l, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int rout_idx = sub2ind_dl(i, k, t, l, num_outer, num_inner, num_trees, num_leaf_nodes_per_tree);
                int distr_idx = sub2ind_dl(t, l, j, 0, num_trees, num_leaf_nodes_per_tree, num_class, 1);
                inter_var_data[inter_idx] = class_label_distr_data[distr_idx] * routing_leaf_data[rout_idx] / 
                 max(tree_prediction_data[pred_idx], Dtype(FLT_MIN));
            }
        }
        for (int n = num_split_nodes_per_tree - 1; n >= 0; n--) {
            int sub_dim_offset = (int) sub_dim_data[sub2ind_dl(t, n, 0, 0, num_trees, num_split_nodes_per_tree, 1, 1)];
            for (int j = 0; j < num_class; ++j) {
                int bottom_idx = sub2ind_dl(i, sub_dim_offset, k/W, k%W, N, C, H, W);
                int inter_idx = sub2ind_dl(i, k, t * num_nodes_pre_tree + n, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int inter_chl_idx = sub2ind_dl(i, k, t * num_nodes_pre_tree + 2 * n + 1, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int inter_chr_idx = sub2ind_dl(i, k, t * num_nodes_pre_tree + 2 * n + 2, j, num_outer, num_inner, num_trees * num_nodes_pre_tree, num_class);
                int label_idx = sub2ind_dl(i, j, k/W, k%W, N, num_class, H, W);
                bottom_diff[bottom_idx] += label_data[label_idx] * (dn_data[bottom_idx] * inter_var_data[inter_chr_idx]
                                - ((Dtype)1.0 - dn_data[bottom_idx]) * inter_var_data[inter_chl_idx]);                          
                inter_var_data[inter_idx] = inter_var_data[inter_chl_idx] + inter_var_data[inter_chr_idx];
            }
        }
    }
}

template <typename Dtype>
__global__ void kernel_backward_dl(Dtype* bottom_diff, Dtype* inter_data, const Dtype* cls_lb_distr, const Dtype* label_data, 
                                   const Dtype* routing_lf, const Dtype* dn_data, const Dtype* tree_pred, const Dtype* dim_offset,
                                   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
                                   int h, int w, int num_classes, int tree_id, int num_dims_) {
  int num_nodes = num_split + num_leaf;
  CUDA_KERNEL_LOOP(index, num_outer) {
    for (int i=0; i<num_inner; ++i) {
      for (int l=0; l<num_leaf; ++l) {
        for (int c=0; c<num_classes; ++c) {
          int inter_idx = sub2ind_dl(index,i,tree_id*num_nodes+num_split+l,c, 
                                 num_outer, num_inner, num_trees*num_nodes, num_classes);
          int cls_lb_distr_idx = sub2ind_dl(tree_id, l, c, 0, num_trees, num_leaf, num_classes, 1);
          int routing_lf_idx = sub2ind_dl(index, i, tree_id, l, num_outer, num_inner, num_trees, num_leaf);
          int tree_pred_idx = sub2ind_dl(index, i, tree_id, c, num_outer, num_inner, num_trees, num_classes);
          inter_data[inter_idx] = cls_lb_distr[cls_lb_distr_idx] * routing_lf[routing_lf_idx] / 
                                 fmaxf(tree_pred[tree_pred_idx], Dtype(FLT_MIN));
        }
      }
      for (int n=num_split-1; n>=0; --n) {
        int dim_offset_idx = sub2ind_dl(tree_id,n,0,0, num_trees,num_split,1,1);
        for (int c=0; c<num_classes; ++c) {
          int lb_idx = sub2ind_dl(index,c,i/w,i%w, num_outer,num_classes,h,w);
          int diff_idx = sub2ind_dl(index,dim_offset[dim_offset_idx],i/w,i%w, num_outer,num_dims_,h,w);
          int inter_left_idx = sub2ind_dl(index,i,tree_id*num_nodes+2*n+1,c,
                                          num_outer,num_inner,num_trees*num_nodes,num_classes);
          int inter_right_idx = inter_left_idx + num_classes;
          const Dtype label_value=label_data[lb_idx];
          bottom_diff[diff_idx] += label_value * (
                    dn_data[diff_idx] * inter_data[inter_right_idx] - 
                    (Dtype(1.0) - dn_data[diff_idx]) * inter_data[inter_left_idx]);
          int inter_parent_idx = sub2ind_dl(index,i,tree_id*num_nodes+n,c,
                                          num_outer,num_inner,num_trees*num_nodes,num_classes);
          inter_data[inter_parent_idx] = inter_data[inter_left_idx] + inter_data[inter_right_idx];
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void kernel_backward_all_dl(Dtype* bottom_diff, Dtype* inter_data, const Dtype* cls_lb_distr, const Dtype* label_data, 
                                   const Dtype* routing_lf, const Dtype* dn_data, const Dtype* tree_pred, const Dtype* dim_offset,
                                   int num_outer, int num_inner, int num_trees, int num_leaf, int num_split, 
                                   int h, int w, int num_classes, int num_dims_) {
  int num_nodes = num_split + num_leaf;
  CUDA_KERNEL_LOOP(index, num_outer) {
    for (int i=0; i<num_inner; ++i) {
      for (int l=0; l<num_leaf; ++l) {
        for(int t= 0; t < num_trees; t++) {
          for (int c=0; c<num_classes; ++c) {
            int inter_idx = sub2ind_dl(index,i,t*num_nodes+num_split+l,c, 
                                   num_outer, num_inner, num_trees*num_nodes, num_classes);
            int cls_lb_distr_idx = sub2ind_dl(t, l, c, 0, num_trees, num_leaf, num_classes, 1);
            int routing_lf_idx = sub2ind_dl(index, i, t, l, num_outer, num_inner, num_trees, num_leaf);
            int tree_pred_idx = sub2ind_dl(index, i, t, c, num_outer, num_inner, num_trees, num_classes);
            inter_data[inter_idx] = cls_lb_distr[cls_lb_distr_idx] * routing_lf[routing_lf_idx] / 
                                   fmaxf(tree_pred[tree_pred_idx], Dtype(FLT_MIN));
          }
        }
      }
      for (int n=num_split-1; n>=0; --n) {
        for(int t = 0; t < num_trees; t++) {
          int dim_offset_idx = sub2ind_dl(t,n,0,0, num_trees,num_split,1,1);
          for (int c=0; c<num_classes; ++c) {
            int lb_idx = sub2ind_dl(index,c,i/w,i%w, num_outer,num_classes,h,w);
            int diff_idx = sub2ind_dl(index,dim_offset[dim_offset_idx],i/w,i%w, num_outer,num_dims_,h,w);
            int inter_left_idx = sub2ind_dl(index,i,t*num_nodes+2*n+1,c,
                                            num_outer,num_inner,num_trees*num_nodes,num_classes);
            int inter_right_idx = inter_left_idx + num_classes;
            const Dtype label_value=label_data[lb_idx];
            bottom_diff[diff_idx] += label_value * (
                      dn_data[diff_idx] * inter_data[inter_right_idx] - 
                      (Dtype(1.0) - dn_data[diff_idx]) * inter_data[inter_left_idx]);
            int inter_parent_idx = sub2ind_dl(index,i,t*num_nodes+n,c,
                                            num_outer,num_inner,num_trees*num_nodes,num_classes);
            inter_data[inter_parent_idx] = inter_data[inter_left_idx] + inter_data[inter_right_idx];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::UpdateTreePredictionAllDataGPU() {
    for (int iter = 0; iter < all_data_vec_length_; iter++) {
        Dtype* tree_prediction_all_data_prob_data = tree_prediction_all_data_prob_vec_[iter].get()->mutable_gpu_data();
        int pred_count = tree_prediction_all_data_prob_vec_[iter].get()->count();
        cudaMemset(tree_prediction_all_data_prob_data, 0, sizeof(Dtype)* pred_count);
        const Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter].get()->gpu_data();
        const Dtype* class_label_distr_data = class_label_distr_->gpu_data();
        int num_outer_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(0);
        int num_inner_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(1);
        CPUTimer timer;
        timer.Start();
        kernel_updata_all_dl<Dtype><<<CAFFE_GET_BLOCKS(pred_count), CAFFE_CUDA_NUM_THREADS>>>(
                  num_outer_iter, num_inner_iter, num_trees_, num_leaf_nodes_per_tree_, num_classes_,
                  routing_leaf_all_data_prob_data, class_label_distr_data, tree_prediction_all_data_prob_data);
        double gpu_time = timer.MicroSeconds()/1000;
        if (debug_gpu_) {
          Dtype* debug_gpu_data = (Dtype*)malloc(sizeof(Dtype)* tree_prediction_all_data_prob_vec_[iter].get()->count());
          memset(debug_gpu_data, 0, sizeof(Dtype)* tree_prediction_all_data_prob_vec_[iter].get()->count());
          const Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter].get()->cpu_data();
          const Dtype* class_label_distr_data = class_label_distr_->cpu_data();
          int num_outer_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(0);
          int num_inner_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(1);
          for (int i = 0; i < num_outer_iter; i++) {
            for (int k = 0; k < num_inner_iter; k++) {
              for (int t = 0; t < num_trees_; t++) {
                caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
                  (Dtype)1.0, routing_leaf_all_data_prob_data + routing_leaf_all_data_prob_vec_[iter].get()->offset(i, k, t, 0), 
                  class_label_distr_data + class_label_distr_->offset(t, 0, 0, 0), (Dtype)0.0, 
                  debug_gpu_data + tree_prediction_all_data_prob_vec_[iter].get()->offset(i, k, t, 0));
              }
            }
          }
          for (int i=0; i<tree_prediction_all_data_prob_vec_[iter].get()->count(); ++i) {
            if (isdiff(debug_gpu_data[i], tree_prediction_all_data_prob_vec_[iter].get()->cpu_data()[i]))
              LOG(FATAL) << "CPU/GPU diff: CPU="<<debug_gpu_data[i]<<", GPU="
                         << tree_prediction_all_data_prob_vec_[iter].get()->cpu_data()[i];
          }
          free(debug_gpu_data);
          LOG(INFO)<<"UpdatePred CPU/GPU check PASS!";
        }
    }
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::UpdateClassLabelDistrGPU() {
  num_epoch_++;
  LOG(INFO) << "Epoch " << num_epoch_ <<": Start updating class label distribution";
  //of_ << "------------------Epoch " << num_epoch_ << " ------------------" << "\n";
  Blob<Dtype> class_label_distr_temp(class_label_distr_->shape());
  Dtype* class_label_distr_temp_data = class_label_distr_temp.mutable_gpu_data();
  int iter_times = 0;
  while (iter_times < iter_times_class_label_distr_) {
      LOG(INFO) << "Label distribution update iteration " << iter_times;
      UpdateTreePredictionAllDataGPU();
      cudaMemset(class_label_distr_temp.mutable_gpu_data(), 0, sizeof(Dtype)* class_label_distr_temp.count());
      // only for CHECK CPU/GPU diff
      Dtype* debug_gpu_data = (Dtype*)malloc(sizeof(Dtype)*class_label_distr_temp.count());
      memset(debug_gpu_data, 0, sizeof(Dtype)* class_label_distr_temp.count());
      for (int iter = 0; iter < all_data_vec_length_; iter++) {
          int num_outer_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(0);
          int num_inner_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(1);
          CPUTimer timer;
          timer.Start();
          kernel_update_leaf_dl<Dtype><<<CAFFE_GET_BLOCKS(num_trees_ * num_leaf_nodes_per_tree_ * num_classes_), CAFFE_CUDA_NUM_THREADS>>>(
            num_trees_, num_leaf_nodes_per_tree_, num_classes_, num_outer_iter, num_inner_iter, 
            class_label_distr_->gpu_data(), all_data_label_vec_[iter].get()->gpu_data(),
            routing_leaf_all_data_prob_vec_[iter].get()->gpu_data(), tree_prediction_all_data_prob_vec_[iter].get()->gpu_data(), 
            class_label_distr_temp.mutable_gpu_data());
          double gpu_time = timer.MicroSeconds()/1000;
          if (debug_gpu_) {
            int num_outer_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(0);
            int num_inner_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(1);
            for (int i = 0; i < num_outer_iter; i++) {
              for (int k = 0; k < num_inner_iter; k++) {
                for (int t = 0; t < num_trees_; t++) {
                  for (int l = 0; l < num_leaf_nodes_per_tree_; l++){
                    for (int j = 0; j < num_classes_; ++j) {
                      debug_gpu_data[class_label_distr_temp.offset(t, l, j, 0)] += all_data_label_vec_[iter].get()->data_at(i, k, j, 0) * 
                      (class_label_distr_->data_at(t, l, j, 0) * routing_leaf_all_data_prob_vec_[iter].get()->data_at(i, k, t, l) 
                        / std::max(tree_prediction_all_data_prob_vec_[iter].get()->data_at(i, k, t, j), Dtype(FLT_MIN)));
                    }
                  }
                }
              }
            }
            for (int i=0; i<class_label_distr_temp.count(); ++i) {
              if (isdiff(class_label_distr_temp.cpu_data()[i],debug_gpu_data[i]))
                LOG(FATAL)<< "CPU/GPU diff:" << "i=" << i <<" CPU="<<debug_gpu_data[i]<< ", GPU=" 
                          << class_label_distr_temp.cpu_data()[i];
            }
            LOG(INFO)<<"Update ClassLabelDistr CPU/GPU check PASS!";
          }
      }
      free(debug_gpu_data);
      memcpy(class_label_distr_->mutable_cpu_data(), class_label_distr_temp.cpu_data(), sizeof(Dtype) * class_label_distr_->count());
      NormalizeClassLabelDistr();
      iter_times++;
  }
  LOG(INFO) << "Epoch" << num_epoch_ << ": End updating class label distribution";
  if (of_.is_open())
      RecordClassLabelDistr();
}
template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    if (!use_gpu_) 
      Forward_cpu(bottom, top);
    else {
      tree_for_training_ = caffe_rng_rand() % num_trees_;
      sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
      const Dtype* dn_data = dn_->gpu_data();
      Dtype* routing_split_prob_data = routing_split_prob_.mutable_gpu_data();
      Dtype* routing_leaf_prob_data = routing_leaf_prob_.mutable_gpu_data();
      const Dtype* sub_dimensions_data = sub_dimensions_->gpu_data();
      kernel_routing_dl<Dtype> << <CAFFE_GET_BLOCKS(num_outer_ * num_inner_ * num_trees_),CAFFE_CUDA_NUM_THREADS >> >(
          num_outer_, num_trees_, num_dims_, bottom[0]->height(), bottom[0]->width(), num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_data, 
      sub_dimensions_data, routing_split_prob_data, routing_leaf_prob_data);
      if (debug_gpu_) {
        Dtype* debug_gpu_rt_leaf = (Dtype*)malloc(sizeof(Dtype)*routing_leaf_prob_.count());
        Dtype* debug_gpu_rt_split = (Dtype*)malloc(sizeof(Dtype)*routing_split_prob_.count());
        for (int i=0; i<routing_split_prob_.count(); i=i+num_split_nodes_per_tree_) {
          debug_gpu_rt_split[i] = Dtype(1.0);
        }
        for (int i=0; i<num_outer_; ++i) {
          for (int k=0; k<num_inner_; ++k) {
            for (int t=0; t<num_trees_; ++t) {
              for (int n=0; n<num_split_nodes_per_tree_; ++n) {
                int current_offset = n;
                int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
                int left_child_offset = 2 * current_offset + 1;
                int right_child_offset = 2 * current_offset + 2;
                if (right_child_offset < num_split_nodes_per_tree_) {
                  debug_gpu_rt_split[routing_split_prob_.offset(i, k, t, left_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width());
                  debug_gpu_rt_split[routing_split_prob_.offset(i, k, t, right_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * ((Dtype) 1.0 - dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width()));
                } else {
                  left_child_offset -= num_split_nodes_per_tree_;
                  right_child_offset -= num_split_nodes_per_tree_;
                  debug_gpu_rt_leaf[routing_leaf_prob_.offset(i, k, t, left_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width());
                  debug_gpu_rt_leaf[routing_leaf_prob_.offset(i, k, t, right_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * ((Dtype) 1.0 - dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width()));
                }
              }
            }
          }
        }
        for (int i=0; i<routing_leaf_prob_.count(); ++i) {
          if (isdiff(debug_gpu_rt_leaf[i],routing_leaf_prob_.cpu_data()[i]))
            LOG(FATAL)<<"CPU/GPU diff: CPU="<<debug_gpu_rt_leaf[i]<<", GPU="<<routing_leaf_prob_.cpu_data()[i];
        }
        for (int i=0; i<routing_split_prob_.count(); ++i) {
          if (isdiff(debug_gpu_rt_split[i],routing_split_prob_.cpu_data()[i]))
            LOG(FATAL)<<"CPU/GPU diff: CPU="<<debug_gpu_rt_split[i]<<", GPU="<<routing_split_prob_.cpu_data()[i];
        }
        free(debug_gpu_rt_leaf);
        free(debug_gpu_rt_split);
        LOG(INFO)<<"Forward routing GPU/CPU check PASS!";
      }
      const Dtype* class_label_distr_data = class_label_distr_->cpu_data();
      Dtype* tree_prediction_prob_data = tree_prediction_prob_.mutable_cpu_data();
      memset(tree_prediction_prob_data, 0, sizeof(Dtype) * tree_prediction_prob_.count());
      Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();
      Dtype* all_data_label_data = all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();
      Dtype loss = (Dtype) 0.0;
      int count = 0;
      for (int i = 0; i < num_outer_; i++) {
          for (int k = 0; k < num_inner_; k++) {
              memcpy(routing_leaf_all_data_prob_data + routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, 0, 0),
              routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, 0, 0), sizeof(Dtype)* num_leaf_nodes_per_tree_ * num_trees_);

              if (drop_out_) {
                caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
                (Dtype)1.0, routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, tree_for_training_, 0),
                class_label_distr_data + class_label_distr_->offset(tree_for_training_, 0, 0, 0),
                (Dtype)0.0, tree_prediction_prob_data + tree_prediction_prob_.offset(i, k, tree_for_training_, 0));
              } else {
                for (int t = 0; t < num_trees_; ++t) {
                  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
                  (Dtype)1.0, routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, t, 0),
                  class_label_distr_data + class_label_distr_->offset(t, 0, 0, 0),
                  (Dtype)0.0, tree_prediction_prob_data + tree_prediction_prob_.offset(i, k, t, 0));
                }
              }
              for(int j = 0; j < num_classes_; ++j) {
                  const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width());
                  DCHECK_GE(label_value, Dtype(0.0)); DCHECK_LE(label_value, Dtype(1.0));
                  all_data_label_data[all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, j, 0)] = label_value;
                  
                if (drop_out_) {
                  loss -= label_value * log(std::max(tree_prediction_prob_.data_at(i, k, tree_for_training_, j), Dtype(FLT_MIN)));
                } else {
                  for(int t = 0; t < num_trees_; t++) {
                    loss -= label_value * log(std::max(tree_prediction_prob_.data_at(i, k, t, j), Dtype(FLT_MIN)));
                  }
                }   
              }
              count++;
          }
      }
      if (std::isnan(loss)) LOG(FATAL)<<"loss nan!";
      top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    }
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) {
  if (!use_gpu_)
    Backward_cpu(top, propagate_down, bottom);
  else {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])  {
      cudaMemset(class_label_distr_->mutable_gpu_diff(), 0, sizeof(Dtype)*class_label_distr_->count());
      cudaMemset(sub_dimensions_->mutable_gpu_diff(), 0, sizeof(Dtype)*sub_dimensions_->count());
      cudaMemset(bottom[0]->mutable_gpu_diff(), 0, sizeof(Dtype)*bottom[0]->count());
      CHECK_EQ(dn_->width(), bottom[1]->width());
      CHECK_EQ(dn_->width(), bottom[1]->width());
      if (iter_times_ + 1 < iter_times_in_epoch_) {
        LOG(INFO) << "iter-time(" << iter_times_ + 1 << ") < iter-times-epoch(" << iter_times_in_epoch_<<
                     "), will pass this backward.";
        iter_times_++;
        return;
      }
      if (drop_out_) {
        kernel_backward_dl<Dtype><<<CAFFE_GET_BLOCKS(num_outer_), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->mutable_gpu_diff(), inter_var_.mutable_gpu_data(), class_label_distr_->gpu_data(),
        bottom[1]->gpu_data(), routing_leaf_prob_.gpu_data(), dn_->gpu_data(),tree_prediction_prob_.gpu_data(), sub_dimensions_->gpu_data(),
        num_outer_, num_inner_,num_trees_, num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_->height(),
        dn_->width(), num_classes_, tree_for_training_, num_dims_);
      } else {
        kernel_backward_all_dl<Dtype><<<CAFFE_GET_BLOCKS(num_outer_), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->mutable_gpu_diff(), inter_var_.mutable_gpu_data(), class_label_distr_->gpu_data(),
        bottom[1]->gpu_data(), routing_leaf_prob_.gpu_data(), dn_->gpu_data(),tree_prediction_prob_.gpu_data(), sub_dimensions_->gpu_data(),
        num_outer_, num_inner_,num_trees_, num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_->height(),
        dn_->width(), num_classes_, num_dims_);
      }
      if (debug_gpu_) {
        Dtype* debug_diff = (Dtype*)malloc(bottom[0]->count()*sizeof(Dtype));
        // caffe_set(class_label_distr_->count(), static_cast<Dtype>(0), class_label_distr_->mutable_cpu_diff());
        // caffe_set(sub_dimensions_->count(), static_cast<Dtype>(0), sub_dimensions_->mutable_cpu_diff());
        caffe_set(bottom[0]->count(), static_cast<Dtype>(0), debug_diff);
        //const Dtype* label = bottom[1]->cpu_data();
        // Dtype* inter_var_data = inter_var_.mutable_cpu_data();
        const Dtype* dn_data = dn_->cpu_data();
        for (int i = 0; i < num_outer_; i++) {
          for (int k = 0; k < num_inner_; k++) {
            if (drop_out_) {
              int t = tree_for_training_;         
              for (int l = 0; l < num_leaf_nodes_per_tree_; l++) {
                for (int j = 0; j < num_classes_; ++j) {
                    Dtype tmp_inter = class_label_distr_->data_at(t, l, j, 0) * routing_leaf_prob_.data_at(i, k, t, l) / 
                                       std::max(tree_prediction_prob_.data_at(i, k, t, j), Dtype(FLT_MIN));
                    if (isdiff(tmp_inter, inter_var_.data_at(i, k, t * num_nodes_per_tree_ + num_split_nodes_per_tree_ + l, j)))
                      LOG(FATAL)<<"inter_var_ CPU/GPU mismatch: CPU="<<tmp_inter<<" GPU="<<
                                inter_var_.data_at(i, k, t * num_nodes_per_tree_ + num_split_nodes_per_tree_ + l, j);
                }
              }
              for (int n = num_split_nodes_per_tree_ - 1; n >= 0; n--) {
                int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
                for (int j = 0; j < num_classes_; ++j) {
                    const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width());
                    debug_diff[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] += label_value *
                    (dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] 
                        * inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 2, j)
                    - ((Dtype)1.0 - dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())]) 
                        * inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 1, j));
                    Dtype tmp_inter_data = inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 2, j) + 
                        inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 1, j); 
                    if (isdiff(tmp_inter_data, inter_var_.data_at(i, k, t * num_nodes_per_tree_ + n, j)))
                      LOG(FATAL)<<"inter_var_ CPU/GPU mismatch: CPU="<<tmp_inter_data<<" GPU="<<
                                inter_var_.data_at(i, k, t * num_nodes_per_tree_ + n, j);
                }
              }
            } else {
              for(int t = 0; t <num_trees_; t++) {         
                for (int l = 0; l < num_leaf_nodes_per_tree_; l++) {
                    for (int j = 0; j < num_classes_; ++j) {
                        Dtype tmp_inter = class_label_distr_->data_at(t, l, j, 0) * routing_leaf_prob_.data_at(i, k, t, l) / 
                                           std::max(tree_prediction_prob_.data_at(i, k, t, j), Dtype(FLT_MIN));
                        if (isdiff(tmp_inter, inter_var_.data_at(i, k, t * num_nodes_per_tree_ + num_split_nodes_per_tree_ + l, j)))
                          LOG(FATAL)<<"inter_var_ CPU/GPU mismatch: CPU="<<tmp_inter<<" GPU="<<
                                    inter_var_.data_at(i, k, t * num_nodes_per_tree_ + num_split_nodes_per_tree_ + l, j);
                    }
                }
                for (int n = num_split_nodes_per_tree_ - 1; n >= 0; n--) {
                  int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
                  for (int j = 0; j < num_classes_; ++j) {
                      const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width());
                      debug_diff[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] += label_value *
                      (dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] 
                          * inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 2, j)
                      - ((Dtype)1.0 - dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())]) 
                          * inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 1, j));
                      Dtype tmp_inter_data = inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 2, j) + 
                          inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 1, j); 
                      if (isdiff(tmp_inter_data, inter_var_.data_at(i, k, t * num_nodes_per_tree_ + n, j)))
                        LOG(FATAL)<<"inter_var_ CPU/GPU mismatch: CPU="<<tmp_inter_data<<" GPU="<<
                                  inter_var_.data_at(i, k, t * num_nodes_per_tree_ + n, j);
                  }
                }
              }
            }
              // check diff
            for (int n=0; n<bottom[0]->count(); ++n) {
              if (isdiff(debug_diff[n], bottom[0]->cpu_diff()[n]))
                LOG(FATAL)<<"diff CPU/GPU mismatch: CPU="<<debug_diff[n]<<" GPU="<<bottom[0]->cpu_diff()[n]<<" index="<<n;
            }
          }
        }
        free(debug_diff);
        LOG(INFO)<<"Backward diff&inter_var check PASS!";
      }
      // Scale down gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(bottom[0]->count(), loss_weight / get_normalizer(normalization_, bottom[0]->count()), bottom[0]->mutable_cpu_diff());
    }
    if (iter_times_ + 1 >= all_data_vec_length_ && (iter_times_+1) % iter_times_in_epoch_ == 0) UpdateClassLabelDistrGPU();
    iter_times_++;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NeuralDecisionDLForestWithLossLayer);

} //end namespace
