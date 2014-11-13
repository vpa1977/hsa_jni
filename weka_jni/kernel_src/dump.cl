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


__kernel void run(__global double* array){
  float lid = get_local_id(0);
   array[ get_global_id(0)+128 ] = isless(lid, 128) ? 128 :  1;
  return;
}
