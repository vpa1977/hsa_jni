/*
 computes min/max values in the range
*/
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
#pragma OPENCL EXTENSION cl_amd_printf : enable
#define REDUCE_STEP_MAX(length, index, width)\
   if (index<width && (index + width)<length){\
      VALUE_TYPE mine = scratch_max[index];\
      VALUE_TYPE other = scratch_max[(index + width)];\
      if (other > mine)\
      {\
      	scratch_max[index] = other;\
      	scratch_index_max[index]  = scratch_index_max[(index + width)];\
      }\
      mine = scratch_min[index];\
      other= scratch_min[(index + width)];\
      if (other < mine)\
      {\
      	scratch_min[index] = other;\
      	scratch_index_min[index]  = scratch_index_min[(index + width)];\
      }\
   }\
   barrier(CLK_LOCAL_MEM_FENCE);

__kernel void run(
   __global VALUE_TYPE *input,
   int length,    
   __global int *result_min,
   __global int *result_max
){

  int local_index = get_local_id(0);
  
  __local VALUE_TYPE scratch_max[GROUP_SIZE], scratch_min[GROUP_SIZE];
  __local int        scratch_index_max[GROUP_SIZE], scratch_index_min[GROUP_SIZE];
  int gx = get_global_id(0);
  int igx_max = gx;
  int igx_min = gx;
  int gloId = gx;
   
  uchar stat;
  
  VALUE_TYPE accumulator_min, accumulator_max;
  if (gloId<length){
     accumulator_max = input[gx];
     accumulator_min = accumulator_max;
     gx = gx + get_global_size(0);
  }
  
  for (; gx<length; gx = gx + get_global_size(0)){
     VALUE_TYPE element = input[gx];
     if (element < accumulator_min) 
     {
     	accumulator_min = element;
     	igx_min = gx;
     }
     if (element > accumulator_max) 
     {
     	accumulator_max = element;
     	igx_max = gx;
     }
     barrier(CLK_LOCAL_MEM_FENCE);
  }

  scratch_max[local_index]  = accumulator_max;
  scratch_index_max[local_index]  = igx_max;
  
  scratch_min[local_index]  = accumulator_min;
  scratch_index_min[local_index]  = igx_min;
  
  barrier(CLK_LOCAL_MEM_FENCE);
  int tail = length - (get_group_id(0) * get_local_size(0));
  REDUCE_STEP_MAX( tail, local_index, 128);
  REDUCE_STEP_MAX( tail, local_index, 64);
  REDUCE_STEP_MAX( tail, local_index, 32);
  REDUCE_STEP_MAX( tail, local_index, 16);
  REDUCE_STEP_MAX( tail, local_index, 8);
  REDUCE_STEP_MAX( tail, local_index, 4);
  REDUCE_STEP_MAX( tail, local_index, 2);
  REDUCE_STEP_MAX( tail, local_index, 1);

  if (local_index==0){
     result_max[get_group_id(0)]  = scratch_index_max[0];
     result_min[get_group_id(0)]  = scratch_index_min[0];
  }
  
  return;
}
