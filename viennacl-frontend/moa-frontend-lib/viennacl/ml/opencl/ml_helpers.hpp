/*
 * weights_update.hpp
 *
 *  Created on: 7/05/2015
 *      Author: bsp
 */

#ifndef VIENNACL_ML_OPENCL_ML_HELPERS_HPP_
#define VIENNACL_ML_OPENCL_ML_HELPERS_HPP_

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/vector_operations.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/ml/opencl/ml_kernels.hpp"
#include "viennacl/ml/opencl/min_max.hpp"
#include "viennacl/ml/opencl/distance.hpp"
#include "viennacl/ml/naive_knn.hpp"

#define KAVERI_GLOBAL_SIZE (4*6+1)*256

namespace viennacl
{
namespace ml
{
namespace opencl
{
	

	template <typename T>
	double reduce( viennacl::vector_base<T>& to_reduce)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(to_reduce).context());
		static int num_groups = (ctx.current_device().max_compute_units() * 4 + 1);
		static int  global_size = num_groups * ctx.current_device().max_work_group_size();

		ml_helper_kernels<viennacl::ocl::context>::init(ctx);
		viennacl::vector<T> reduce_result(num_groups, ctx);
		std::vector<T> reduce_result_cpu(num_groups);
		static viennacl::ocl::kernel& reduce = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "reduce");
		reduce.local_work_size(0,ctx.current_device().max_work_group_size());
		reduce.global_work_size(0, global_size);
		viennacl::ocl::enqueue(reduce(to_reduce.size(), viennacl::ocl::local_mem(reduce.local_work_size(0) * sizeof(cl_double)),   to_reduce, reduce_result));
		viennacl::copy(reduce_result, reduce_result_cpu);
		double ret = 0;
		for (auto val : reduce_result_cpu)
			ret += val;
		return ret;

	}

	template <typename T>
	void sgd_compute_factors(const viennacl::vector_base<T>& classes, const viennacl::vector_base<T>& prod_result, viennacl::vector_base<T>& factors, const bool is_nominal, int loss, double learning_rate, double bias)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(classes).context());

		ml_helper_kernels<viennacl::ocl::context>::init(ctx);
		viennacl::ocl::kernel& sgd_map_prod_value = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "sgd_map_prod_value");
		static size_t  global_size = (ctx.current_device().max_compute_units() *4 +1) * ctx.current_device().max_work_group_size();
		sgd_map_prod_value.local_work_size(0, ctx.current_device().max_work_group_size());
		sgd_map_prod_value.global_work_size(0, global_size);


		//(int N, bool nominal,int loss, double learning_rate, double bias, __global double* class_values, __global double* prod_values, __global double* factor)
		cl_ulong size = classes.size();
		viennacl::ocl::enqueue(sgd_map_prod_value(size, (cl_uint)is_nominal,(cl_uint) loss, (cl_double) learning_rate,  (cl_double)bias, classes.handle().opencl_handle(), prod_result.handle().opencl_handle(),
				factors.handle().opencl_handle()
				));

	};

	template<typename T= double> 
	void dense_sgd_update_weights(int row, bool nominal, double learning_rate, double bias, unsigned int loss_function, const viennacl::vector<double>& class_values, const viennacl::scalar<double>& prod_result, const viennacl::vector<double>& row_values, viennacl::vector<double>& weights)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(weights).context());
		ml_helper_kernels<viennacl::ocl::context>::init(ctx);
		static bool init = false;
		viennacl::ocl::kernel& dense_sgd_map_prod_values_kernel = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "dense_sgd_map_prod_value");
		static int global_size = (ctx.current_device().max_compute_units() * 4 + 1) * ctx.current_device().max_work_group_size();
		dense_sgd_map_prod_values_kernel.local_work_size(0, ctx.current_device().max_work_group_size());
		dense_sgd_map_prod_values_kernel.global_work_size(0, global_size);
		viennacl::ocl::enqueue(dense_sgd_map_prod_values_kernel(
			row_values.size(), // rows
			nominal,
			loss_function,
			row,
			learning_rate,
			bias,
			class_values.handle().opencl_handle(),
			prod_result,
			weights.handle().opencl_handle(), 
			weights.handle().opencl_handle()
			));
		
	}

	template <typename sgd_matrix_type, typename T =double>
	void sgd_update_weights(viennacl::vector_base<T>& weights, const sgd_matrix_type& batch, const viennacl::vector_base<T>& factors)
	{
		viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(weights).context());
		ml_helper_kernels<viennacl::ocl::context>::init(ctx);
		viennacl::ocl::kernel& update_by_factor_kernel = ctx.get_kernel(ml_helper_kernels<viennacl::ocl::context>::program_name(), "sgd_update_weights");
		static size_t global_size = (ctx.current_device().max_compute_units() *4 +1) * ctx.current_device().max_work_group_size();
		update_by_factor_kernel.local_work_size(0, ctx.current_device().max_work_group_size());
		size_t work_size = batch.size1();
		if (ctx.current_device().max_work_group_size() >(size_t) work_size)
			work_size = ctx.current_device().max_work_group_size();
		if (work_size > global_size)
			work_size = global_size;
		update_by_factor_kernel.global_work_size(0, work_size);

		/*
                 int columns = batch.size2();
		std::cout << "Factor " << factors(0) << std::endl;
		
		for (int i = 0; i < batch.size2(); ++i)
		{
			double d = ((sgd_matrix_type&)batch)(0, i);
			std::cout << " " << d;
		}
		std::cout << std::endl;
		*/
		viennacl::ocl::enqueue(update_by_factor_kernel(
				batch.size1(), // rows
				batch.handle().opencl_handle(), // data
				factors.handle().opencl_handle(), // factors vector
				batch.handle1().opencl_handle(), // row indices vector 
				batch.handle2().opencl_handle(), // column indices vector
				weights.handle().opencl_handle() // weights vector
				));

/*		std::cout << "Weights after:";
		for (int i = 0; i < weights.size(); ++i)
		{
			std::cout << " " << weights(i);

		}
		std::cout << std::endl;
		std::cout << "matrix after mult ";
		for (int i = 0; i < batch.size2(); ++i)
		{
			double d = ((sgd_matrix_type&)batch)(0, i);
			std::cout << " " << d;
		}
		std::cout << std::endl;
*/
	};

	struct knn_kernels
	{
		template <typename NumericT>
		static void min_max(size_t num_attributes, size_t class_index, viennacl::vector<NumericT>& source, viennacl::vector<NumericT>& min_val, viennacl::vector<NumericT>& max_val)
		{
			viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(source).context());
			min_max_kernel<NumericT, viennacl::ocl::context>::init(ctx);
			static viennacl::ocl::kernel& min_max = ctx.get_kernel(min_max_kernel<NumericT, viennacl::ocl::context>::program_name(), "min_max_kernel");
			size_t global_size = num_attributes * 256;
			min_max.local_work_size(0, 256);
			min_max.global_work_size(0, global_size);
			viennacl::ocl::enqueue(min_max((int)class_index, (int)num_attributes, (int)(source.size() / num_attributes), source, min_val, max_val));
		}

		template <typename NumericT>
		static void distance_1(size_t num_attributes, const viennacl::vector<NumericT>&  input, const viennacl::vector<NumericT>&  samples,
			const viennacl::vector<NumericT>&  range_min, const viennacl::vector<NumericT>&  range_max, 
			viennacl::vector<NumericT>&  range_min_upd, viennacl::vector<NumericT>&  range_max_upd,
			const viennacl::vector<int>&  attribute_map, viennacl::vector<NumericT>&  result)
		{
			viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input).context());
			distance_kernel<NumericT, viennacl::ocl::context>::init(ctx);
			static viennacl::ocl::kernel& distance_1 = ctx.get_kernel(distance_kernel<NumericT, viennacl::ocl::context>::program_name(), "square_distance");
			static viennacl::ocl::kernel& bounds = ctx.get_kernel(distance_kernel<NumericT, viennacl::ocl::context>::program_name(), "update_bounds");
			assert(samples.size() > 0);
			const size_t min_local_size = 64;
			size_t local_size = 256;
			size_t work_size = samples.size() / num_attributes;
			if  (work_size % local_size)
				work_size = (work_size / local_size) * local_size + local_size;
			distance_1.local_work_size(0, local_size);
			distance_1.global_work_size(0, work_size);
			
			local_size = std::min((size_t)256, num_attributes);
			work_size = num_attributes;
			if (local_size %  min_local_size)
				local_size = (local_size / min_local_size) * min_local_size + min_local_size;
			if (work_size % local_size)
				work_size = (work_size / local_size) * local_size + local_size;
			bounds.local_work_size(0,local_size);
			bounds.global_work_size(0,work_size);

			bounds((int)num_attributes, range_min, range_max, input, range_min_upd, range_max_upd).enqueue();
			distance_1(input, samples, range_min_upd, range_max_upd, attribute_map, result, (int) (samples.size() / num_attributes), (int)num_attributes).enqueue();
		}

		template <typename NumericT>
		static void distance_2(size_t num_attributes, const viennacl::vector<NumericT>&  input, const viennacl::vector<NumericT>&  samples,
			const viennacl::vector<NumericT>&  range_min, const viennacl::vector<NumericT>&  range_max, 
			viennacl::vector<NumericT>&  range_min_upd, viennacl::vector<NumericT>&  range_max_upd,
			const viennacl::vector<int>&  attribute_map, viennacl::vector<NumericT>&  result)
		{
			viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input).context());
			const size_t min_local_size = 64;
			assert(samples.size() > 0);
			size_t local_size = std::min(num_attributes, (size_t) 256);
			if (local_size % min_local_size)
				local_size = (local_size / min_local_size) * min_local_size + min_local_size;
			viennacl::ocl::local_mem scratch(local_size);
			size_t work_size = (samples.size() / num_attributes) * local_size;
			distance_kernel<NumericT, viennacl::ocl::context>::init(ctx);
			static viennacl::ocl::kernel& distance_2 = ctx.get_kernel(distance_kernel<NumericT, viennacl::ocl::context>::program_name(), "square_distance_one_wg");
			static viennacl::ocl::kernel& bounds = ctx.get_kernel(distance_kernel<NumericT, viennacl::ocl::context>::program_name(), "update_bounds");
			distance_2.local_work_size(0, local_size);
			distance_2.global_work_size(0, work_size);

			local_size = std::min((size_t)256, num_attributes);
			work_size = num_attributes;
			if (local_size %  min_local_size)
				local_size = (local_size / min_local_size) * min_local_size + min_local_size;
			if (work_size % local_size)
				work_size = (work_size / local_size) * local_size + local_size;
			bounds.local_work_size(0, local_size);
			bounds.global_work_size(0, work_size);
			bounds((int)num_attributes, range_min, range_max, input, range_min_upd, range_max_upd).enqueue();

			distance_2(input, samples, range_min_upd, range_max_upd, attribute_map, result, (int)(samples.size() / num_attributes),(int) num_attributes, scratch ).enqueue();
		}


	};
}

}
}



#endif /* VIENNACL_ML_WEIGHTS_UPDATE_HPP_ */
