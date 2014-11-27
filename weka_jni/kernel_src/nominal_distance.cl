
							   
__kernel void run(__global const double* temp,
							  const int len,
							  __global double* result)
{
	int id = get_global_id(0);
	int offset = id * len;
	double val = temp[offset];
	for (int i = 1 ;i < len ; ++i) 
		val += temp[i+offset];
	result[id] = val;	
}
	
