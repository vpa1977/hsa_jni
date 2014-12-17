#define TYPE double
#define userFunctor(x,y) (x)*(y)

  
  
 __kernel void run( 
 	global TYPE* input_iter,
 	global uint* indices,
 	global TYPE* input_iter2,
 	const int len, 
 	global TYPE* result
 ) 
 {
 	TYPE value1 = input_iter[get_global_id(1) * len + get_global_id(0)];
 	TYPE value2 = input_iter2[indices[get_global_id(0)]];
 	result [get_global_id(1) * len + get_global_id(0)] = userFunctor(value1 , value2); 
 }; 


