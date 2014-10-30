/***************************************************************************
*  Based on bolt::cl::max_element function: 
* 
*   © 2012,2014 Advanced Micro Devices, Inc. All rights reserved.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/
#define VALUE_TYPE double
#define GROUP_SIZE 256

#define REDUCE_STEP_MAX(length, index, width)\
   if (index<width && (index + width)<length){\
      VALUE_TYPE mine = scratch[index];\
      VALUE_TYPE other = scratch[(index + width)];\
      scratch[index]  = (other<mine)?mine:other;\
      scratch_index[index]  = (other<mine)?scratch_index[index]:scratch_index[(index + width)];\
   }


__kernel void run(
   __global VALUE_TYPE *input,
   int length,    
  // int stride, 
  // int offset, 
   __global int *result, 
   __global double* output
){
  int local_index = get_local_id(0);
  int stride = 1;int offset =0;
  __local VALUE_TYPE scratch[GROUP_SIZE];
  __local int scratch_index[GROUP_SIZE];

  int gx = get_global_id(0);
  int igx = gx;
  int gloId = gx; 
  bool stat;
  
  VALUE_TYPE accumulator;
  if (gloId<length){
     accumulator = input[((gx * stride) + offset)];
     gx = gx + get_global_size(0);
  }
  for (; gx<length; gx = gx + get_global_size(0)){
     VALUE_TYPE element = input[((gx * stride) + offset)];
     stat = (element<accumulator)?1:0;
     accumulator = (stat)?accumulator:element;
     igx = (stat)?igx:gx;
  }

  scratch[local_index]  = accumulator;
  scratch_index[local_index]  = igx;
  barrier(CLK_LOCAL_MEM_FENCE);
  int tail = length - (get_group_id(0) * get_local_size(0));
 // REDUCE_STEP_MAX( tail, local_index, 128);
  barrier(CLK_LOCAL_MEM_FENCE);
 // REDUCE_STEP_MAX( tail, local_index, 64);
  barrier(CLK_LOCAL_MEM_FENCE);
//  REDUCE_STEP_MAX( tail, local_index, 32);
  barrier(CLK_LOCAL_MEM_FENCE);
//  REDUCE_STEP_MAX( tail, local_index, 16);
  barrier(CLK_LOCAL_MEM_FENCE);

  if (local_index<8 && (local_index + 8)<length){
      VALUE_TYPE mine = scratch[local_index];
      VALUE_TYPE other = scratch[(local_index + 8)];
      if (other > mine) 
      {
	scratch[local_index]  =other;
	scratch_index[local_index] = scratch_index[local_index+8];
      }
   }

  //REDUCE_STEP_MAX( tail, local_index, 8);
  barrier(CLK_LOCAL_MEM_FENCE);
//  REDUCE_STEP_MAX( tail, local_index, 4);
  barrier(CLK_LOCAL_MEM_FENCE);
 // REDUCE_STEP_MAX( tail, local_index, 2);
  barrier(CLK_LOCAL_MEM_FENCE);
 // REDUCE_STEP_MAX( tail, local_index, 1);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (gloId>=length){
     return;
  }

  result[get_global_id(0)] = scratch_index[local_index];
  //if (local_index==0){
  //   result[get_group_id(0)]  = scratch_index[0];
  //}
  result[33] = length;
  return;
}
