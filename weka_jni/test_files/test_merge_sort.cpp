/*
 * test_max_ele.cpp
 *
 *  Created on: 23/10/2014
 *      Author: bsp
 */


#include "max_value.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wcomment"

#include <bolt/cl/inner_product.h>
#include <bolt/cl/count.h>
#include <bolt/cl/device_vector.h>
#include <bolt/cl/scan.h>
#include <bolt/cl/scatter.h>
#include <bolt/cl/transform.h>
#include <bolt/cl/max_element.h>

#include "merge_sort.hpp"


#pragma GCC diagnostic pop
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define BILLION 1000000000L
int main()
{
	HSAContext* context = HSAContext::Create();

	std::vector<double> major_test;
	for (int i = 0 ;i < 1024*1024; ++i)
		major_test.push_back(1024*1024 -i);
	//major_test[1024] = (double)RAND_MAX + (double)100;
	major_test[33] = -22;
	{
		MergeSort sort(context,"/home/bsp/hsa_jni/kernels/local_merge.brig","/home/bsp/hsa_jni/kernels/global_merge.brig" );
		std::vector<int> offsets;

		//sort.sort(major_test, offsets);
		//printf("result %f\n", major_test[0]);
		offsets.resize(major_test.size());

		double max;
		timespec ts, ts1;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		for (int i = 0 ; i < 1 ; ++i)
			sort.sort(major_test, offsets);
		clock_gettime(CLOCK_MONOTONIC, &ts1);
		uint64_t	 diff = BILLION * (ts1.tv_sec - ts.tv_sec) + ts1.tv_nsec - ts.tv_nsec;
		printf("HSA elapsed time = %llu msec/10\n", (long long unsigned int) (diff/100000));
/*
		bolt::cl::device_vector<double> dev(major_test.begin(), major_test.end());
		clock_gettime(CLOCK_MONOTONIC, &ts);
		for (int i = 0 ; i < 1000 ; ++i)
			 max = *(bolt::cl::max_element(dev.begin(), dev.end()));
		clock_gettime(CLOCK_MONOTONIC, &ts1);
		diff = BILLION * (ts1.tv_sec - ts.tv_sec) + ts1.tv_nsec - ts.tv_nsec;
		printf("BOLT elapsed time = %llu msec\n", (long long unsigned int) (diff/1000000));


		clock_gettime(CLOCK_MONOTONIC, &ts);
		for (int i = 0 ; i < 1000 ; ++i)
			 max = *(std::max_element(major_test.begin(), major_test.end()));
		clock_gettime(CLOCK_MONOTONIC, &ts1);
		diff = BILLION * (ts1.tv_sec - ts.tv_sec) + ts1.tv_nsec - ts.tv_nsec;
		printf("SEQ elapsed time = %llu msec\n", (long long unsigned int) (diff/1000000));
		printf("Max > RAND_MAX %d", max > RAND_MAX);
*/
	}


}

