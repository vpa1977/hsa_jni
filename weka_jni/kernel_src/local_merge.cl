#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define GROUP_SIZE 256

uint lowerBoundBinarylocal(__local double* data, uint left, uint right, double searchVal){
   uint firstIndex = left;
   uint lastIndex = right;
   for (; firstIndex<lastIndex; ){
      uint midIndex = (firstIndex + lastIndex) / 2;
      double midValue = data[midIndex];
      
      if (midValue < searchVal){
         firstIndex = midIndex + 1;
      } else {
         lastIndex = midIndex;
      }
   }
   return(firstIndex);
}
uint upperBoundBinarylocal(__local double* data, uint left, uint right, double searchVal){
   uint upperBound = lowerBoundBinarylocal( data, left, right, searchVal);
   if (upperBound!=right){
      uint mid = 0;
      for (double upperValue = data[upperBound]; ( upperValue == searchVal) && upperBound<right; upperValue = data[upperBound]){
         mid = (upperBound + right) / 2;
         double midValue = data[mid];
         if (midValue == searchVal){
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
   uint vecSize, 
   __global double *data_iter, 
   __global uint* offsets
   
){
	__local double lds[GROUP_SIZE];
	__local double lds2[GROUP_SIZE];
	
	__local double indices[GROUP_SIZE];
	__local double indices2[GROUP_SIZE];
  { 
 	size_t gloId = get_global_id( 0 ); 
 	size_t groId = get_group_id( 0 ); 
 	size_t locId = get_local_id( 0 ); 
 	size_t wgSize = get_local_size( 0 ); 
 
 
 
 	double val; 
	 if( gloId < vecSize) 
	 { 
	 	val = data_iter[ gloId ]; 
	 	lds[locId ] = val; 
	 	indices[locId] = gloId;
	 } 
	 barrier( CLK_LOCAL_MEM_FENCE ); 
 
 	uint end = wgSize; 
 	if( (groId+1)*(wgSize) >= vecSize ) 
 		end = vecSize - (groId*wgSize); 
 
 	uint numMerges = 8; 
 	uint pass; 
 	for( pass = 1; pass <= numMerges; ++pass ) 
 	{ 
 		uint srcLogicalBlockSize = 1 << (pass-1); 
 		if( gloId < vecSize) 
 		{ 
 			uint srcBlockNum = (locId) / srcLogicalBlockSize; 
 			uint srcBlockIndex = (locId) % srcLogicalBlockSize; 
 
 			uint dstLogicalBlockSize = srcLogicalBlockSize<<1; 
 			uint leftBlockIndex = (locId) & ~(dstLogicalBlockSize - 1 ); 
 
 			leftBlockIndex += (srcBlockNum & 0x1) ? 0 : srcLogicalBlockSize; 
 			leftBlockIndex = (((leftBlockIndex) < (end)) ? (leftBlockIndex) : (end)); 
 			uint rightBlockIndex = (((leftBlockIndex + srcLogicalBlockSize) < (end)) ? (leftBlockIndex + srcLogicalBlockSize) : (end)); 
 
			uint insertionIndex = 0; 
 			if(pass%2 != 0) 
 			{ 
 				if( (srcBlockNum & 0x1) == 0 ) 
 				{ 
 					insertionIndex = lowerBoundBinarylocal( lds, leftBlockIndex, rightBlockIndex, lds[ locId ]) - leftBlockIndex; 
 				} 
 				else 
 				{ 
 					insertionIndex = upperBoundBinarylocal( lds, leftBlockIndex, rightBlockIndex, lds[ locId ]) - leftBlockIndex; 
 				} 
 			} 
 			else 
 			{ 
 				if( (srcBlockNum & 0x1) == 0 ) 
 				{ 
 					insertionIndex = lowerBoundBinarylocal( lds2, leftBlockIndex, rightBlockIndex, lds2[ locId ] ) - leftBlockIndex; 
 				} 
 				else 
 				{ 
 					insertionIndex = upperBoundBinarylocal( lds2, leftBlockIndex, rightBlockIndex, lds2[ locId ] ) - leftBlockIndex; 
 				} 
 			} 
 			uint dstBlockIndex = srcBlockIndex + insertionIndex; 
 			uint dstBlockNum = srcBlockNum/2;
 			uint offset =(dstBlockNum*dstLogicalBlockSize)+dstBlockIndex; 
 			if(pass%2 != 0)
 			{ 
 				 
 				lds2[offset ] = lds[ locId ];
 				indices2[offset] = indices[locId];
 			} 
 			else 
 			{
 				lds[ (dstBlockNum*dstLogicalBlockSize)+dstBlockIndex ] = lds2[ locId ];
 				indices[offset] = indices2[locId];
 			} 
 		} 
 		barrier( CLK_LOCAL_MEM_FENCE ); 
 	} 
 	if( gloId < vecSize) 
 	{ 
 		val = lds[ locId ]; 
 		data_iter[ gloId ] = val;
 		offsets[gloId] = indices[locId]; 
 	} 
 } 
}

