

uint lowerBoundBinary(__global double* source_iter, uint left, uint right, double searchVal){
   uint firstIndex = left;
   uint lastIndex = right;
   for (; firstIndex<lastIndex; ){
      uint midIndex = (firstIndex + lastIndex) / 2;
      double midValue = source_iter[midIndex];
      
      if (midValue < searchVal){
         firstIndex = midIndex + 1;
      } else {
         lastIndex = midIndex;
      }
   }
   return(firstIndex);
}

uint upperBoundBinary( __global double* source_iter, uint left, uint right, double searchVal){
   uint upperBound = lowerBoundBinary( source_iter, left, right, searchVal);
   if (upperBound!=right){
      uint mid = 0;
      for (double upperValue = source_iter[upperBound]; searchVal ==  upperValue && upperBound<right; upperValue = source_iter[upperBound]){
         mid = (upperBound + right) / 2;
         double midValue = source_iter[mid];
         if (midValue ==  searchVal){
            upperBound = mid + 1;
         } else {
            right = mid;
            upperBound++;
         }
      }
   }
   return(upperBound);
}
__kernel void run(
   uint srcVecSize, 
   uint srcLogicalBlockSize, 
   __global double *source_iter, 
   __global double *result_iter, 
   __global uint* src_offsets, 
   __global uint* result_offsets
)
{
	size_t globalID = get_global_id( 0 ); 
 	size_t groupID = get_group_id( 0 ); 
 	size_t localID = get_local_id( 0 ); 
 	size_t wgSize = get_local_size( 0 ); 
 
  
 	if( globalID >= srcVecSize ) 
 		return; 
 	uint srcBlockNum = globalID / srcLogicalBlockSize; 
 	uint srcBlockIndex = globalID % srcLogicalBlockSize; 
 
 
 	uint dstLogicalBlockSize = srcLogicalBlockSize<<1; 
 	uint leftBlockIndex = globalID & ~(dstLogicalBlockSize - 1 ); 
 
 	leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize; 
 	leftBlockIndex = (((leftBlockIndex) < (srcVecSize)) ? (leftBlockIndex) : (srcVecSize)); 
 	uint rightBlockIndex = (((leftBlockIndex + srcLogicalBlockSize) < (srcVecSize)) ? (leftBlockIndex + srcLogicalBlockSize) : (srcVecSize)); 
 
 	uint insertionIndex = 0; 
 	double search_val = source_iter[ globalID ]; 
 	if( (srcBlockNum & 0x1) == 0 ) 
 	{ 
 		insertionIndex = lowerBoundBinary( source_iter, leftBlockIndex, rightBlockIndex, search_val ) - leftBlockIndex; 
 	} 
 	else 
 	{ 
 		insertionIndex = upperBoundBinary( source_iter, leftBlockIndex, rightBlockIndex, search_val ) - leftBlockIndex; 
 	} 
 
 	uint dstBlockIndex = srcBlockIndex + insertionIndex; 
 	uint dstBlockNum = srcBlockNum/2; 
 	uint offset = (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex; 
 	result_iter[ offset ] = source_iter[ globalID ]; 
 	result_offsets[offset ] = src_offsets[ globalID];

}