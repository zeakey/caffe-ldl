#ifndef CAFFE_NEURAL_DECISION_FOREST_LAYER_HPP_
#define CAFFE_NEURAL_DECISION_FOREST_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe
{
	template <typename Dtype>
	class NeuralDecisionForestLayer : public Layer<Dtype>
	{
	public:
		explicit NeuralDecisionForestLayer(const LayerParameter& param)
			: Layer<Dtype>(param), sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
			dn_(new Blob<Dtype>()) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "NeuralDecisionForest"; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
		void InitRoutingProb();
		void RecordClassLabelDistr();

		int num_trees_;
		int num_classes_;
		int depth_;

		int num_split_nodes_per_tree_;
		int num_leaf_nodes_per_tree_;
		int num_nodes_pre_tree_;

		int num_outer_;
		int num_inner_;

		int num_nodes_;
		int num_dims_;

		int axis_;


		/// bottom vector holder used in call to the underlying SigmoidLayer::Forward (fn)
		vector<Blob<Dtype>*> sigmoid_bottom_vec_;
		/// top vector holder used in call to the underlying SigmoidLayer::Forward (dn)
		vector<Blob<Dtype>*> sigmoid_top_vec_;
		/// the probabilities of sending samples to left subtrees
		shared_ptr<Blob<Dtype> > dn_;
		/// the probabilities that each sample falls into a split node (\mu)
		Blob<Dtype> routing_split_prob_;
		/// the probabilities that each sample falls into a leaf node (\mu)
		Blob<Dtype> routing_leaf_prob_;
		/// the class label distribution of each leaf node
		/// It does not actually hosts the blobs (blobs_ does), so we simply store pointers.
		Blob<Dtype>* class_label_distr_;
		/// the dimensions used to train each tree
		/// It does not actually hosts the blobs (blobs_ does), so we simply store pointers.
		Blob<Dtype>* sub_dimensions_;
		/// the predicted probabilities of each sample given by forest
		Blob<Dtype> forest_prediction_prob_;

		/// output of the network
		//Blob<Dtype> output_prob_;
		///the file to save class distributions
		std::ofstream of_;
	};
}
#endif