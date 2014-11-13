
							   
__kernel void run(__global const double* temp,
							  const int len,
							  __global double* result)
{
	int id = get_global_id(0);
	double val = temp[0];
	for (int i = 1 ;i < len ; ++i) 
		val += temp[i];
	result[id] = val;	
}
	
