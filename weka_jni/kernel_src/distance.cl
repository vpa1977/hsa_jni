#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/*
   distance kernel one
*/
__kernel void square_distance(__global const double* input,
						__global const double* samples,
						__global const double2* ranges,
						__global double* result,
						const int window_size, 
						const int element_count, 
						const int numerics_size)
{     
	   
	int result_offset = get_global_id(0);
	if (result_offset >= window_size) // handle last group
		return;
		
	int vector_offset = element_count * result_offset;
	double point_distance = 0;
	double val;
	double width;
	int i;
	for (i = 0; i < numerics_size ; i ++ ) 
	{
		double2 range = ranges[i];
		width = ( range.y - range.x);
		val = width > 0 ? (input[i] - range.x) / width  - (samples[ vector_offset + i] - range.x)/width : 0;
		point_distance += val*val; 
	}
	
	for (; i < element_count; i ++ ) 
	{
		point_distance += isnotequal( input[i] , samples[vector_offset + i]);
	}
	result[result_offset] = point_distance;
}         



