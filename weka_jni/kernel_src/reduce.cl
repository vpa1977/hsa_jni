#define TYPE double
#define userFunctor(x,y) (x)+(y)

 #define _REDUCE_STEP(_LENGTH, _IDX, _W)\
	 if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
 		TYPE mine = scratch[_IDX];\
 		TYPE other = scratch[_IDX + _W];\
 		scratch[_IDX] = userFunctor(mine, other);\
 	  }\
      barrier(CLK_LOCAL_MEM_FENCE);
 
  
 __kernel void run( 
 	global TYPE* input_iter, 
 	const int length, 
 	global TYPE* result
 ) 
 { 
 	local TYPE[256] scratch;
 	int gx = get_global_id (0); 
 	int gloId = gx; 
 	TYPE accumulator; 
 	if(gloId < length){ 
 		accumulator = (TYPE) input_iter[gx]; 
 		gx += get_global_size(0); 
 	} 
 
	 while (gx < length) 
	 { 
	 	TYPE element = input_iter[gx]; 
	 	accumulator = userFunctor(accumulator, element); 
	 	gx += get_global_size(0); 
	 } 
	 
 	int local_index = get_local_id(0); 
 	scratch[local_index] = accumulator; 
 	barrier(CLK_LOCAL_MEM_FENCE); 
 
 	uint tail = length - (get_group_id(0) * get_local_size(0)); 
 
 	_REDUCE_STEP(tail, local_index, 128); 
 	_REDUCE_STEP(tail, local_index, 64); 
 	_REDUCE_STEP(tail, local_index, 32); 
 	_REDUCE_STEP(tail, local_index, 16); 
 	_REDUCE_STEP(tail, local_index, 8); 
 	_REDUCE_STEP(tail, local_index, 4); 
 	_REDUCE_STEP(tail, local_index, 2); 
 	_REDUCE_STEP(tail, local_index, 1); 
 
 	if( gloId >= length ) 
 		return; 
 
 	if (local_index == 0) { 
 		result[get_group_id(0)] = scratch[0]; 
 	} 
 }; 


