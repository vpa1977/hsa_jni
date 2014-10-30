/*
 * test_hsa_context_main.cpp
 *
 *  Created on: 23/10/2014
 *      Author: bsp
 */
#include <string.h>
#include "HSAContext.h"

int main()
{
	double max_finder[10] = {1,2,3,4,5,6,7,8,9};
	int  result[512];
	memset(result,0, sizeof(result));
	int second_guy[512];
	memset(result,0, sizeof(result));
	double holy_molly[512];
	memset(holy_molly,0, sizeof(holy_molly));

	HSAContext* context = HSAContext::Create();
	//context->registerArrayMemory((void*)max_finder, (int) sizeof(max_finder));
	//context->registerArrayMemory((void*)result, (int) sizeof(result));
	//context->registerArrayMemory((void*)second_guy, (int) sizeof(second_guy));
	//context->registerArrayMemory((void*)holy_molly, (int) sizeof(holy_molly));

	HSAContext::Kernel* kernel = context->createKernel("/home/bsp/hsa_jni/kernels/max_value.brig", "&__OpenCL_run_kernel");
	HSAContext::Dispatch* dispatch = context->createDispatch(kernel);

	dispatch->pushPointerArg(0);
	dispatch->pushPointerArg(0);
	dispatch->pushPointerArg(0);
	dispatch->pushPointerArg(0);
	dispatch->pushPointerArg(0);
	dispatch->pushPointerArg(0);

	dispatch->pushPointerArg((void*)max_finder);
	dispatch->pushIntArg((int)(sizeof(max_finder)/sizeof(double)) );
	dispatch->pushIntArg(1);
	dispatch->pushIntArg(0);
	dispatch->pushPointerArg((void*)result);
	//dispatch->pushPointerArg((void*)holy_molly);
	size_t global_dims[3] = {256,1,1};
	size_t local_dims[3] = {256,1,1};
	dispatch->setLaunchAttributes(1, global_dims,  local_dims);
	dispatch->dispatchKernelWaitComplete();
	printf("done and result[0] is %f\n", max_finder[result[0]]);

	HSAContext::Kernel* kernel2 = context->createKernel("/home/bsp/hsa_jni/kernels/max_value.brig", "&__OpenCL_run_kernel");
	HSAContext::Dispatch* dispatch2 = context->createDispatch(kernel2);
	delete dispatch;
	delete kernel;


	dispatch2->clearArgs();
	dispatch2->pushPointerArg(0);
	dispatch2->pushPointerArg(0);
	dispatch2->pushPointerArg(0);
	dispatch2->pushPointerArg(0);
	dispatch2->pushPointerArg(0);
	dispatch2->pushPointerArg(0);

	dispatch2->pushPointerArg((void*)max_finder);
	dispatch2->pushIntArg((int)(sizeof(max_finder)/sizeof(double)) );
	dispatch2->pushIntArg(1);
	dispatch2->pushIntArg(0);
	dispatch2->pushPointerArg((void*)result);
	dispatch2->setLaunchAttributes(1, global_dims,  local_dims);
	max_finder[0] = 2334;
	dispatch2->dispatchKernelWaitComplete();
	printf("done and result[0] is %f\n", max_finder[result[0]]);
	delete dispatch2;
	delete kernel2;

	context->dispose();
}



