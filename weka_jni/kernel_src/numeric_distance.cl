#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable


__kernel void run(__global const double* samples,
							     double* test_value,
							  const double* range, 
							  const int window_size,
							  const int numerics,
							  const int instance_size,
							  __global double* result)
{
	int window_pos = get_global_id(0);
	int attribute = get_global_id(1);
	int offset = attribute * window_size + window_pos;
	double range_min = range[ attribute * 2];
	double range_max = range[ attribute *2 +1];
	double width = (range_max - range_min);
	double val = 0; 
	if (attribute < numerics)
		val = width > 0 ? (test_value[attribute] - range_min) / width  - (samples[offset] - range_min)/width : 0;
	else
		val = samples[offset] - test_value[attribute] != 0 ? 1 : 0;
	result[window_pos*instance_size + attribute] = val*val;
}