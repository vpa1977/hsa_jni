// University of Illinois/NCSA
// Open Source License
// 
// Copyright (c) 2013, Advanced Micro Devices, Inc.
// All rights reserved.
// 
// Developed by:
// 
//     Runtimes Team
// 
//     Advanced Micro Devices, Inc
// 
//     www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal with
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// 
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimers.
// 
//     * Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimers in the
//       documentation and/or other materials provided with the distribution.
// 
//     * Neither the names of the LLVM Team, University of Illinois at
//       Urbana-Champaign, nor the names of its contributors may be used to
//       endorse or promote products derived from this Software without specific
//       prior written permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
// SOFTWARE.
//===----------------------------------------------------------------------===//

#ifndef HSACONTEXT_H
#define HSACONTEXT_H
#include <future>
#include <hsa.h>


#ifdef __cplusplus
#define _CPPSTRING_ "C"
#endif
#ifndef __cplusplus
#define _CPPSTRING_
#endif
#ifndef __SNACK_DEFS
typedef struct transfer_t Transfer_t;
struct transfer_t { int nargs ; size_t* rsrvd1; size_t* rsrvd2 ; size_t* rsrvd3; } ;
typedef struct lparm_t Launch_params_t;
struct lparm_t { int ndim; size_t gdims[3]; size_t ldims[3]; Transfer_t transfer  ;} ;
#define __SNACK_DEFS
#endif

#define FIX_ARGS_STABLE(X)\
	X->pushPointerArg(0);\
	X->pushPointerArg(0);\
	X->pushPointerArg(0);\
	X->pushPointerArg(0);\
	X->pushPointerArg(0);\
	X->pushPointerArg(0);


// Abstract interface to an HSA Implementation
class HSAContext {
public:

	/** hardware-specific. hardcoded for simplicity */
	virtual size_t GetComputeUnits() =0;
	virtual size_t GetMaxWorkgroup()=0;
	virtual ~HSAContext() {}
public:
	class Dispatch {
	public:
		virtual ~Dispatch() {};
		// various methods for setting different types of args into the arg stack
		virtual hsa_status_t pushFloatArg(float) = 0;
		virtual hsa_status_t pushIntArg(int) = 0;
		virtual hsa_status_t pushBooleanArg(unsigned char) = 0;
		virtual hsa_status_t pushByteArg(char) = 0;
		virtual hsa_status_t pushLongArg(long) = 0;
		virtual hsa_status_t pushDoubleArg(double) = 0;
		virtual hsa_status_t pushPointerArg(void *addr) = 0;
		virtual hsa_status_t clearArgs() = 0;
		// allow a previously pushed arg to be changed
		virtual hsa_status_t setPointerArg(int idx, void *addr) = 0;

		// setting number of dimensions and sizes of each
		virtual hsa_status_t setLaunchAttributes(int dims, size_t *globalDims,
				size_t *localDims) = 0;

		// run a kernel and wait until complete
		virtual hsa_status_t dispatchKernelWaitComplete() = 0;

		// dispatch a kernel asynchronously
		virtual hsa_status_t dispatchKernel(hsa_signal_t& ) = 0;

		// wait for the kernel to finish execution
		virtual hsa_status_t waitComplete(hsa_signal_t& ) = 0;

		// dispatch a kernel asynchronously and get a future object
		virtual std::future<void> dispatchKernelAndGetFuture() = 0;
	};

	class Kernel {
	public:
		virtual ~Kernel() {};
	};

	// create a kernel object from the specified HSAIL text source and entrypoint
	virtual Kernel* createKernel(const char *source, const char *entryName) = 0;

	// create a kernel dispatch object from the specified kernel
	virtual Dispatch* createDispatch(const Kernel *kernel) = 0;

	// dispose of an environment including all programs
	virtual hsa_status_t dispose() = 0;

	virtual hsa_status_t registerArrayMemory(void *addr, int lengthInBytes) = 0;

	static HSAContext* Create();

private:
	static HSAContext* m_pContext;
};



#endif // HSACONTEXT_H
