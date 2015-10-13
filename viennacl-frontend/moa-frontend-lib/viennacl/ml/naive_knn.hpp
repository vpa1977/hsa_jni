#ifndef VIENNACL_ML_NAIVEKNN_HPP
#define VIENNACL_ML_NAIVEKNN_HPP
#include <viennacl/forwards.h>
#include <viennacl/context.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/maxmin.hpp>
#include <viennacl/matrix.hpp>
#include <memory>



#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/ml/opencl/ml_helpers.hpp"
#include "viennacl/ml/opencl/merge_sort.hpp"
#endif

namespace viennacl
{
namespace ml
{
namespace knn
{
	enum AttributeType {
		atNUMERIC, 
		atNOMINAL
	};

	enum DistanceWeighting {
		dwINVERSE,
		dwSIMILARITY, 
		dwNONE
	};

	template <typename NumericType>
	class normalizable_distance
	{
	public:
		normalizable_distance(viennacl::context& ctx, size_t class_attribute, size_t attribute_size)
			: class_attribute_(class_attribute),
			attribute_size_(attribute_size),
			ctx_(ctx),
			max_values_upd_(attribute_size, ctx),
			min_values_upd_(attribute_size, ctx),
			max_values_(attribute_size, ctx),
			min_values_(attribute_size, ctx), 
			attribute_map_(attribute_size, ctx)
		{
		}

		void set_attribute_map(const viennacl::vector<int>& map)
		{
			attribute_map_ = map;
		}

		void set(const viennacl::vector<NumericType>& data)
		{
			data_ = data;
			switch (ctx_.memory_type())
			{
			case OPENCL_MEMORY:
				viennacl::ml::opencl::knn_kernels::min_max<NumericType>(attribute_size_, class_attribute_, data_, min_values_, max_values_);
				break;
			case MAIN_MEMORY:
				compute_min_max_cpu();
				break;
			default:
				throw memory_exception("Not implemented");
			}
		}
		size_t num_instances()
		{
			return data_.size() / attribute_size_;
		}
		size_t num_attributes_used()
		{
			return attribute_size_ - 1;
		}
		virtual void distance(const viennacl::vector<NumericType>& point, viennacl::vector<NumericType>& distances) {};
	private:
		void compute_min_max_cpu()
		{
			size_t num_instances = data_.size() / attribute_size_;
			for (int attribute = 0; attribute < attribute_size_; ++attribute)
			{
				min_values_[attribute] = std::numeric_limits<NumericType>::max();
				max_values_[attribute] = std::numeric_limits<NumericType>::min();
			}
			for (int inst = 0; inst < num_instances; ++inst)
			{
				for (int attribute = 0; attribute < attribute_size_; ++attribute)
				{
					size_t index = inst* attribute_size_ + attribute;
					if (data_[index] > max_values_[attribute])
						max_values_[attribute] = data_[index];
					if (data_[index] < min_values_[attribute])
						min_values_[attribute] = data_[index];
				}
			}
		}
	protected:
		viennacl::context& ctx_;
		size_t class_attribute_;
		size_t attribute_size_;
		viennacl::vector<NumericType> data_;
		viennacl::vector<NumericType> max_values_upd_;
		viennacl::vector<NumericType> min_values_upd_;
		viennacl::vector<NumericType> max_values_;
		viennacl::vector<NumericType> min_values_;
		viennacl::vector<int> attribute_map_;
	};

	template <typename NumericType> 
	class euclidean_distance : public normalizable_distance<NumericType>
	{
	public:
		euclidean_distance(viennacl::context& ctx, size_t class_attribute, size_t attribute_size) :
			normalizable_distance(ctx, class_attribute, attribute_size)
		{}
		virtual void distance(const viennacl::vector<NumericType>& point, viennacl::vector<NumericType>& distances) 
		{
				switch (ctx_.memory_type())
				{
				case viennacl::MAIN_MEMORY:
					calc_distance_cpu(distances, point);
					break;
#ifdef VIENNACL_WITH_OPENCL
				case viennacl::OPENCL_MEMORY:
					viennacl::ml::opencl::knn_kernels::distance_1(attribute_size_, 
						point, data_, 
						min_values_, max_values_, 
						min_values_upd_, max_values_upd_,
						attribute_map_, distances);
					break;
#endif
#ifdef VIENNACL_WITH_HSA
				case viennacl::HSA_MEMORY:
					throw memory_exception("Not implemented");
					break;
#endif
				default:
					throw memory_exception("Not implemented");
			}
		}
	protected:
		void calc_distance_cpu(viennacl::vector<double>& result, const viennacl::vector<double>& sample)
		{
			update_bounds(sample);
			size_t num_instances = data_.size() / attribute_size_;
			for (int instance = 0; instance < num_instances; ++instance)
			{
				result(instance) = point_distance(instance, sample);
			}
				
		}

		void update_bounds(const viennacl::vector<double>& sample)
		{
			for (int i = 0; i < attribute_size_; ++i)
			{
				double value = sample(i);
				max_values_upd_(i) = value > max_values_(i) ? value : max_values_[i];
				min_values_upd_(i) = value < min_values_(i) ? value : min_values_[i];
			}
		}

		double point_distance(int instance, const viennacl::vector<double>& sample)
		{
			double distance = 0;
			size_t offset = instance * attribute_size_;
			for (size_t pos = offset; pos < offset + attribute_size_; ++pos)
			{
				size_t attribute = pos - offset;
			//	std::cout << "Processing " << attribute << " value " << sample(attribute) << "min " << min_values_(attribute) << " max " << max_values_(attribute) << " data " << data_(pos) << std::endl;
				if (attribute == class_attribute_)
					continue;
				double sample_norm = (sample(attribute) - min_values_upd_(attribute)) / (max_values_upd_(attribute) - min_values_upd_(attribute));
				double example_norm = (data_(pos) - min_values_upd_(attribute)) / (max_values_upd_(attribute) - min_values_upd_(attribute));
				double d = (sample_norm - example_norm);
				if (attribute_map_(attribute) == atNOMINAL)
					d = d != 0 ? 1 : 0;
				
				distance += d * d;
			}
			return distance;
		}
	};

	template <typename NumericType>
	class euclidean_distance_wg : public euclidean_distance<NumericType>
	{
	public:
		euclidean_distance_wg(viennacl::context& ctx, size_t class_attribute, size_t attribute_size) :
			euclidean_distance(ctx, class_attribute, attribute_size)
		{}
		virtual void distance(const viennacl::vector<NumericType>& point, viennacl::vector<NumericType>& distances)
		{
			switch (ctx_.memory_type())
			{
			case viennacl::MAIN_MEMORY:
				calc_distance_cpu(distances, point);
				break;
#ifdef VIENNACL_WITH_OPENCL
			case viennacl::OPENCL_MEMORY:
				viennacl::ml::opencl::knn_kernels::distance_2(attribute_size_,
					point, data_,
					min_values_, max_values_,
					min_values_upd_, max_values_upd_,
					attribute_map_, distances);

				break;
#endif
#ifdef VIENNACL_WITH_HSA
			case viennacl::HSA_MEMORY:
				throw memory_exception("Not implemented");
				break;
#endif
			default:
				throw memory_exception("Not implemented");
			}
		}
	};

	/** 
	*/
	template <typename NumericType>
	class sort_strategy
	{
	public:
		sort_strategy(bool need_distances) : need_distances_(need_distances) {}
		virtual void set(size_t size, const viennacl::context& ctx) = 0;
		virtual void sort(viennacl::vector<NumericType>& distances) = 0;
		std::vector<int>& indices() { return cpu_indices_; }
		std::vector<NumericType> distances() { assert(need_distances_); return cpu_distances_; }
	protected:
		void cpu_sort(viennacl::vector<NumericType>& distances)
		{
			for (size_t i = 0; i != cpu_indices_.size(); ++i) cpu_indices_[i] = (int)i;
			viennacl::copy(distances, cpu_distances_);
			std::sort(cpu_indices_.begin(), cpu_indices_.end(),
				[this](size_t i1, size_t i2) {return cpu_distances_[i1] < cpu_distances_[i2]; });
			if (need_distances_)
				std::sort(cpu_distances_.begin(), cpu_distances_.end());
		}
	protected:
		bool need_distances_;
		std::vector<int> cpu_indices_;
		std::vector<NumericType> cpu_distances_;

	};

	template <typename NumericType> 
	class default_sort_strategy : public sort_strategy<NumericType>
	{
	public:
		default_sort_strategy(bool need_distances) : sort_strategy(need_distances) {}
		void set(size_t size, const viennacl::context& ctx)
		{
			cpu_indices_.resize(size);
			if (ctx.memory_type() == MAIN_MEMORY || need_distances_)
				cpu_distances_.resize(size);
			if (ctx.memory_type() == OPENCL_MEMORY)
				opencl_sorter_ = merge_sorter<NumericType>(size, ctx);
		}

		void sort(viennacl::vector<NumericType>& distances)
		{
			const viennacl::context& ctx = viennacl::traits::context(distances);
			switch (ctx.memory_type())
			{
			case OPENCL_MEMORY:
				viennacl::copy(opencl_sorter_.merge_sort(distances), cpu_indices_);
				if (need_distances_)
					viennacl::copy(distances, cpu_distances_);
				break;
			case MAIN_MEMORY:
				cpu_sort(distances);
			break;
			default:
				throw memory_exception("Not implemented");
			}
		}
	private:
		merge_sorter<NumericType> opencl_sorter_;
	};

	struct knn_options
	{
		DistanceWeighting distance_weighting_;
		AttributeType class_type_;
		size_t num_classes_;
		size_t class_index_;
		size_t k_;
	};

	template <typename NumericType>
	class naive_knn
	{
	public:
		naive_knn(const knn_options& options, 
			const viennacl::vector<int>& attribute_map, 
			normalizable_distance<NumericType>& distance_function, 
			sort_strategy<NumericType>& sorter,
			viennacl::context& ctx) :
			context_(ctx),
			attribute_map_(attribute_map), 
			distance_function_(distance_function), 
			sort_strategy_(sorter), 
			options_(options)
		{}

		void train(const viennacl::vector<NumericType>& data, const std::vector<NumericType> class_values,const std::vector<NumericType> weights)
		{
			distance_function_.set(data);
			distances_ = viennacl::vector<NumericType>(data.size(), viennacl::traits::context(data));
			class_values_ = class_values;
			weights_ = weights;
		}

		std::vector<double> distribution_for_instance(const viennacl::vector<NumericType>& test)
		{
			std::vector<double> result;
			distance_function_.distance(test, distances_);
			sort_strategy_.sort(distances_);
			//
			double total = 0, weight;
			std::vector<int>& cpu_indices = sort_strategy_.indices();
			std::vector<NumericType>& cpu_distance = sort_strategy_.distances();
			std::vector<double> distribution(options_.num_classes_);
			/*WEKA distributionForInstance()*/
			// Set up a correction to the estimator
			if (options_.num_classes_ == atNOMINAL)
			{
				for (int i = 0; i < options_.num_classes_; i++)
					distribution[i] = 1.0f / (1.0f> distance_function_.num_instances() ? 1.0f : distance_function_.num_instances());
				total = (double)options_.num_classes_ / (1 > distance_function_.num_instances() ? 1 : distance_function_.num_instances());
			}

			for (int i = 0; i < options_.k_; i++) {
				// Collect class counts
				int index = cpu_indices.at(i);
				if (options_.distance_weighting_ != dwNONE)
				{
					cpu_distance[i] = std::sqrt(cpu_distance[i] / distance_function_.num_attributes_used());
					if (options_.distance_weighting_ == dwINVERSE)
						weight = 1.0 / (cpu_distance[i] + 0.001); // to avoid div by zero
					if (options_.distance_weighting_ == dwSIMILARITY)
							weight = 1.0 - cpu_distance[i];
				}
				else
					weight = 1.0;
				weight *= weights_.at(index);
				
				if (options_.class_type_ == atNOMINAL)
					distribution[(int)class_values_.at(index)] += weight;
				if (options_.class_type_ == atNUMERIC)
					distribution[0] += class_values_.at(index) * weight;
				total += weight;
			}

			// Normalise distribution
			if (total > 0) {
				for (int i = 0; i < distribution.size(); ++i)
					distribution[i] = distribution[i] / total;
			}
			return distribution;
		}


	private:
		viennacl::context& context_;
		viennacl::vector<int> attribute_map_;
		viennacl::vector<NumericType> distances_;
		normalizable_distance<NumericType>& distance_function_;
		sort_strategy<NumericType>& sort_strategy_;
		knn_options options_;
		std::vector<NumericType> class_values_;
		std::vector<NumericType> weights_;
	};


	/**
	 * KD-Tree design follows WEKA KD-tree
	 */
	struct kd_tree_node
	{
	};

	class kd_tree_node_splitter
	{
	public:
		virtual void split(kd_tree_node& parent) = 0;
	};


	



}
}
}


#endif
