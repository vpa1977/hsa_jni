

// #define VALUE_TYPE double
enum   AttributeType
{
		atNUMERIC, 
		atNOMINAL
};
/*
   distance kernel one
*/
__kernel void square_distance(__global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const VALUE_TYPE* range_min,
						__global const VALUE_TYPE* range_max,
						__global const int* attribute_type,
						__global VALUE_TYPE* result,
						const int length, 
						const int attribute_size)
{     
	   
	int result_offset = get_global_id(0);
	if (result_offset >= length) // handle last group
		return;
		
	int vector_offset = attribute_size * result_offset;
	VALUE_TYPE point_distance = 0;
	VALUE_TYPE val;
	VALUE_TYPE width;
	int i;
	for (i = 0; i < attribute_size ; i ++ ) 
	{
		if (attribute_type[i] == atNUMERIC) 
		{
			width = ( range_max[i] - range_min[i]);
			val = width > 0 ? (input[i] - range_min[i]) / width  - (samples[ vector_offset + i] - range_min[i])/width : 0;
			point_distance += val*val; 
		}
		else
		{
			point_distance += isnotequal( input[i] , samples[vector_offset + i]);
		}
	}
	result[result_offset] = point_distance;
}         



