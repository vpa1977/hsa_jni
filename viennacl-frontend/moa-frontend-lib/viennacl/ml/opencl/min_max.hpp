#ifndef VIENNACL_ML_OPENCL_MIN_MAX_HPP_
#define VIENNACL_ML_OPENCL_MIN_MAX_HPP_

#include <viennacl/ocl/context.hpp>

namespace viennacl
{
	namespace ml
	{
		namespace opencl
		{
			template <typename NumericT, class context_class>
			struct min_max_kernel
			{

				static std::string program_name()
				{
					return "knn_min_max";
				}

				static void init(context_class& ctx)
				{
					static bool done = false; // TODO : multiple hsa contexts. At the moment there can be only one
					if (done)
						return;
					done = true;
					std::string code;
					std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();
					code.append("#define VALUE_TYPE ");
					code.append(numeric_string);
					code.append("\n");
					
					/* from min_max.cl*/
					unsigned char min_max_cl[] = {
						0x2f, 0x2a, 0x0d, 0x0a, 0x20, 0x63, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65,
						0x73, 0x20, 0x6d, 0x69, 0x6e, 0x2f, 0x6d, 0x61, 0x78, 0x20, 0x76, 0x61,
						0x6c, 0x75, 0x65, 0x73, 0x20, 0x69, 0x6e, 0x20, 0x74, 0x68, 0x65, 0x20,
						0x72, 0x61, 0x6e, 0x67, 0x65, 0x0d, 0x0a, 0x2a, 0x2f, 0x0d, 0x0a, 0x2f,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x0d, 0x0a, 0x2a, 0x20, 0x20, 0x42, 0x61, 0x73, 0x65,
						0x64, 0x20, 0x6f, 0x6e, 0x20, 0x62, 0x6f, 0x6c, 0x74, 0x3a, 0x3a, 0x63,
						0x6c, 0x3a, 0x3a, 0x6d, 0x61, 0x78, 0x5f, 0x65, 0x6c, 0x65, 0x6d, 0x65,
						0x6e, 0x74, 0x20, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x3a,
						0x20, 0x0d, 0x0a, 0x2a, 0x20, 0x0d, 0x0a, 0x2a, 0x20, 0x20, 0x20, 0xa9,
						0x20, 0x32, 0x30, 0x31, 0x32, 0x2c, 0x32, 0x30, 0x31, 0x34, 0x20, 0x41,
						0x64, 0x76, 0x61, 0x6e, 0x63, 0x65, 0x64, 0x20, 0x4d, 0x69, 0x63, 0x72,
						0x6f, 0x20, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x73, 0x2c, 0x20, 0x49,
						0x6e, 0x63, 0x2e, 0x20, 0x41, 0x6c, 0x6c, 0x20, 0x72, 0x69, 0x67, 0x68,
						0x74, 0x73, 0x20, 0x72, 0x65, 0x73, 0x65, 0x72, 0x76, 0x65, 0x64, 0x2e,
						0x0d, 0x0a, 0x2a, 0x0d, 0x0a, 0x2a, 0x20, 0x20, 0x20, 0x4c, 0x69, 0x63,
						0x65, 0x6e, 0x73, 0x65, 0x64, 0x20, 0x75, 0x6e, 0x64, 0x65, 0x72, 0x20,
						0x74, 0x68, 0x65, 0x20, 0x41, 0x70, 0x61, 0x63, 0x68, 0x65, 0x20, 0x4c,
						0x69, 0x63, 0x65, 0x6e, 0x73, 0x65, 0x2c, 0x20, 0x56, 0x65, 0x72, 0x73,
						0x69, 0x6f, 0x6e, 0x20, 0x32, 0x2e, 0x30, 0x20, 0x28, 0x74, 0x68, 0x65,
						0x20, 0x22, 0x4c, 0x69, 0x63, 0x65, 0x6e, 0x73, 0x65, 0x22, 0x29, 0x3b,
						0x0d, 0x0a, 0x2a, 0x20, 0x20, 0x20, 0x79, 0x6f, 0x75, 0x20, 0x6d, 0x61,
						0x79, 0x20, 0x6e, 0x6f, 0x74, 0x20, 0x75, 0x73, 0x65, 0x20, 0x74, 0x68,
						0x69, 0x73, 0x20, 0x66, 0x69, 0x6c, 0x65, 0x20, 0x65, 0x78, 0x63, 0x65,
						0x70, 0x74, 0x20, 0x69, 0x6e, 0x20, 0x63, 0x6f, 0x6d, 0x70, 0x6c, 0x69,
						0x61, 0x6e, 0x63, 0x65, 0x20, 0x77, 0x69, 0x74, 0x68, 0x20, 0x74, 0x68,
						0x65, 0x20, 0x4c, 0x69, 0x63, 0x65, 0x6e, 0x73, 0x65, 0x2e, 0x0d, 0x0a,
						0x2a, 0x20, 0x20, 0x20, 0x59, 0x6f, 0x75, 0x20, 0x6d, 0x61, 0x79, 0x20,
						0x6f, 0x62, 0x74, 0x61, 0x69, 0x6e, 0x20, 0x61, 0x20, 0x63, 0x6f, 0x70,
						0x79, 0x20, 0x6f, 0x66, 0x20, 0x74, 0x68, 0x65, 0x20, 0x4c, 0x69, 0x63,
						0x65, 0x6e, 0x73, 0x65, 0x20, 0x61, 0x74, 0x0d, 0x0a, 0x2a, 0x0d, 0x0a,
						0x2a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x68, 0x74, 0x74, 0x70,
						0x3a, 0x2f, 0x2f, 0x77, 0x77, 0x77, 0x2e, 0x61, 0x70, 0x61, 0x63, 0x68,
						0x65, 0x2e, 0x6f, 0x72, 0x67, 0x2f, 0x6c, 0x69, 0x63, 0x65, 0x6e, 0x73,
						0x65, 0x73, 0x2f, 0x4c, 0x49, 0x43, 0x45, 0x4e, 0x53, 0x45, 0x2d, 0x32,
						0x2e, 0x30, 0x0d, 0x0a, 0x2a, 0x0d, 0x0a, 0x2a, 0x20, 0x20, 0x20, 0x55,
						0x6e, 0x6c, 0x65, 0x73, 0x73, 0x20, 0x72, 0x65, 0x71, 0x75, 0x69, 0x72,
						0x65, 0x64, 0x20, 0x62, 0x79, 0x20, 0x61, 0x70, 0x70, 0x6c, 0x69, 0x63,
						0x61, 0x62, 0x6c, 0x65, 0x20, 0x6c, 0x61, 0x77, 0x20, 0x6f, 0x72, 0x20,
						0x61, 0x67, 0x72, 0x65, 0x65, 0x64, 0x20, 0x74, 0x6f, 0x20, 0x69, 0x6e,
						0x20, 0x77, 0x72, 0x69, 0x74, 0x69, 0x6e, 0x67, 0x2c, 0x20, 0x73, 0x6f,
						0x66, 0x74, 0x77, 0x61, 0x72, 0x65, 0x0d, 0x0a, 0x2a, 0x20, 0x20, 0x20,
						0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x20,
						0x75, 0x6e, 0x64, 0x65, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x4c, 0x69,
						0x63, 0x65, 0x6e, 0x73, 0x65, 0x20, 0x69, 0x73, 0x20, 0x64, 0x69, 0x73,
						0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x20, 0x6f, 0x6e, 0x20,
						0x61, 0x6e, 0x20, 0x22, 0x41, 0x53, 0x20, 0x49, 0x53, 0x22, 0x20, 0x42,
						0x41, 0x53, 0x49, 0x53, 0x2c, 0x0d, 0x0a, 0x2a, 0x20, 0x20, 0x20, 0x57,
						0x49, 0x54, 0x48, 0x4f, 0x55, 0x54, 0x20, 0x57, 0x41, 0x52, 0x52, 0x41,
						0x4e, 0x54, 0x49, 0x45, 0x53, 0x20, 0x4f, 0x52, 0x20, 0x43, 0x4f, 0x4e,
						0x44, 0x49, 0x54, 0x49, 0x4f, 0x4e, 0x53, 0x20, 0x4f, 0x46, 0x20, 0x41,
						0x4e, 0x59, 0x20, 0x4b, 0x49, 0x4e, 0x44, 0x2c, 0x20, 0x65, 0x69, 0x74,
						0x68, 0x65, 0x72, 0x20, 0x65, 0x78, 0x70, 0x72, 0x65, 0x73, 0x73, 0x20,
						0x6f, 0x72, 0x20, 0x69, 0x6d, 0x70, 0x6c, 0x69, 0x65, 0x64, 0x2e, 0x0d,
						0x0a, 0x2a, 0x20, 0x20, 0x20, 0x53, 0x65, 0x65, 0x20, 0x74, 0x68, 0x65,
						0x20, 0x4c, 0x69, 0x63, 0x65, 0x6e, 0x73, 0x65, 0x20, 0x66, 0x6f, 0x72,
						0x20, 0x74, 0x68, 0x65, 0x20, 0x73, 0x70, 0x65, 0x63, 0x69, 0x66, 0x69,
						0x63, 0x20, 0x6c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x20, 0x67,
						0x6f, 0x76, 0x65, 0x72, 0x6e, 0x69, 0x6e, 0x67, 0x20, 0x70, 0x65, 0x72,
						0x6d, 0x69, 0x73, 0x73, 0x69, 0x6f, 0x6e, 0x73, 0x20, 0x61, 0x6e, 0x64,
						0x0d, 0x0a, 0x2a, 0x20, 0x20, 0x20, 0x6c, 0x69, 0x6d, 0x69, 0x74, 0x61,
						0x74, 0x69, 0x6f, 0x6e, 0x73, 0x20, 0x75, 0x6e, 0x64, 0x65, 0x72, 0x20,
						0x74, 0x68, 0x65, 0x20, 0x4c, 0x69, 0x63, 0x65, 0x6e, 0x73, 0x65, 0x2e,
						0x0d, 0x0a, 0x0d, 0x0a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a,
						0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2f, 0x0d, 0x0a, 0x23, 0x64,
						0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x47, 0x52, 0x4f, 0x55, 0x50, 0x5f,
						0x53, 0x49, 0x5a, 0x45, 0x20, 0x32, 0x35, 0x36, 0x0d, 0x0a, 0x23, 0x70,
						0x72, 0x61, 0x67, 0x6d, 0x61, 0x20, 0x4f, 0x50, 0x45, 0x4e, 0x43, 0x4c,
						0x20, 0x45, 0x58, 0x54, 0x45, 0x4e, 0x53, 0x49, 0x4f, 0x4e, 0x20, 0x63,
						0x6c, 0x5f, 0x61, 0x6d, 0x64, 0x5f, 0x70, 0x72, 0x69, 0x6e, 0x74, 0x66,
						0x20, 0x3a, 0x20, 0x65, 0x6e, 0x61, 0x62, 0x6c, 0x65, 0x0d, 0x0a, 0x23,
						0x64, 0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x52, 0x45, 0x44, 0x55, 0x43,
						0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d, 0x49, 0x4e, 0x5f, 0x4d,
						0x41, 0x58, 0x28, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x2c, 0x20, 0x69,
						0x6e, 0x64, 0x65, 0x78, 0x2c, 0x20, 0x77, 0x69, 0x64, 0x74, 0x68, 0x29,
						0x5c, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x69, 0x66, 0x20, 0x28, 0x69, 0x6e,
						0x64, 0x65, 0x78, 0x3c, 0x77, 0x69, 0x64, 0x74, 0x68, 0x20, 0x26, 0x26,
						0x20, 0x28, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x20, 0x2b, 0x20, 0x77, 0x69,
						0x64, 0x74, 0x68, 0x29, 0x3c, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x29,
						0x7b, 0x5c, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x56, 0x41,
						0x4c, 0x55, 0x45, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x20, 0x6d, 0x69, 0x6e,
						0x65, 0x20, 0x3d, 0x20, 0x73, 0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f,
						0x6d, 0x61, 0x78, 0x5b, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x5d, 0x3b, 0x5c,
						0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x56, 0x41, 0x4c, 0x55,
						0x45, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x20, 0x6f, 0x74, 0x68, 0x65, 0x72,
						0x20, 0x3d, 0x20, 0x73, 0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d,
						0x61, 0x78, 0x5b, 0x28, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x20, 0x2b, 0x20,
						0x77, 0x69, 0x64, 0x74, 0x68, 0x29, 0x5d, 0x3b, 0x5c, 0x0d, 0x0a, 0x20,
						0x20, 0x20, 0x20, 0x20, 0x20, 0x69, 0x66, 0x20, 0x28, 0x6f, 0x74, 0x68,
						0x65, 0x72, 0x20, 0x3e, 0x20, 0x6d, 0x69, 0x6e, 0x65, 0x29, 0x5c, 0x0d,
						0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x7b, 0x5c, 0x0d, 0x0a, 0x20,
						0x20, 0x20, 0x20, 0x20, 0x20, 0x09, 0x73, 0x63, 0x72, 0x61, 0x74, 0x63,
						0x68, 0x5f, 0x6d, 0x61, 0x78, 0x5b, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x5d,
						0x20, 0x3d, 0x20, 0x6f, 0x74, 0x68, 0x65, 0x72, 0x3b, 0x5c, 0x0d, 0x0a,
						0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x7d, 0x5c, 0x0d, 0x0a, 0x20, 0x20,
						0x20, 0x20, 0x20, 0x20, 0x6d, 0x69, 0x6e, 0x65, 0x20, 0x3d, 0x20, 0x73,
						0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d, 0x69, 0x6e, 0x5b, 0x69,
						0x6e, 0x64, 0x65, 0x78, 0x5d, 0x3b, 0x5c, 0x0d, 0x0a, 0x20, 0x20, 0x20,
						0x20, 0x20, 0x20, 0x6f, 0x74, 0x68, 0x65, 0x72, 0x3d, 0x20, 0x73, 0x63,
						0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d, 0x69, 0x6e, 0x5b, 0x28, 0x69,
						0x6e, 0x64, 0x65, 0x78, 0x20, 0x2b, 0x20, 0x77, 0x69, 0x64, 0x74, 0x68,
						0x29, 0x5d, 0x3b, 0x5c, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
						0x69, 0x66, 0x20, 0x28, 0x6f, 0x74, 0x68, 0x65, 0x72, 0x20, 0x3c, 0x20,
						0x6d, 0x69, 0x6e, 0x65, 0x29, 0x5c, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
						0x20, 0x20, 0x7b, 0x5c, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
						0x09, 0x73, 0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d, 0x69, 0x6e,
						0x5b, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x5d, 0x20, 0x3d, 0x20, 0x6f, 0x74,
						0x68, 0x65, 0x72, 0x3b, 0x5c, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20,
						0x20, 0x7d, 0x5c, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x7d, 0x5c, 0x0d, 0x0a,
						0x20, 0x20, 0x20, 0x62, 0x61, 0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43,
						0x4c, 0x4b, 0x5f, 0x4c, 0x4f, 0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d,
						0x5f, 0x46, 0x45, 0x4e, 0x43, 0x45, 0x29, 0x3b, 0x0d, 0x0a, 0x0d, 0x0a,
						0x2f, 0x2a, 0x2a, 0x20, 0x0d, 0x0a, 0x09, 0x73, 0x63, 0x61, 0x6e, 0x20,
						0x6d, 0x69, 0x6e, 0x2f, 0x6d, 0x61, 0x78, 0x20, 0x76, 0x61, 0x6c, 0x75,
						0x65, 0x2e, 0x20, 0x0d, 0x0a, 0x09, 0x31, 0x20, 0x77, 0x6f, 0x72, 0x6b,
						0x67, 0x72, 0x6f, 0x75, 0x70, 0x20, 0x70, 0x65, 0x72, 0x20, 0x61, 0x74,
						0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x0d, 0x0a, 0x09, 0x6f, 0x75,
						0x74, 0x70, 0x75, 0x74, 0x20, 0x31, 0x20, 0x65, 0x6e, 0x74, 0x72, 0x79,
						0x20, 0x70, 0x65, 0x72, 0x20, 0x77, 0x6f, 0x72, 0x6b, 0x67, 0x72, 0x6f,
						0x75, 0x70, 0x0d, 0x0a, 0x2a, 0x2f, 0x0d, 0x0a, 0x5f, 0x5f, 0x6b, 0x65,
						0x72, 0x6e, 0x65, 0x6c, 0x20, 0x76, 0x6f, 0x69, 0x64, 0x20, 0x6d, 0x69,
						0x6e, 0x5f, 0x6d, 0x61, 0x78, 0x5f, 0x6b, 0x65, 0x72, 0x6e, 0x65, 0x6c,
						0x28, 0x0d, 0x0a, 0x09, 0x69, 0x6e, 0x74, 0x20, 0x63, 0x6c, 0x61, 0x73,
						0x73, 0x5f, 0x61, 0x74, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x2c,
						0x0d, 0x0a, 0x09, 0x69, 0x6e, 0x74, 0x20, 0x73, 0x74, 0x72, 0x69, 0x64,
						0x65, 0x2c, 0x20, 0x2f, 0x2a, 0x20, 0x6e, 0x75, 0x6d, 0x62, 0x65, 0x72,
						0x20, 0x6f, 0x66, 0x20, 0x61, 0x74, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74,
						0x65, 0x73, 0x20, 0x2a, 0x2f, 0x0d, 0x0a, 0x09, 0x69, 0x6e, 0x74, 0x20,
						0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x2c, 0x20, 0x2f, 0x2a, 0x20, 0x6e,
						0x75, 0x6d, 0x62, 0x65, 0x72, 0x20, 0x6f, 0x66, 0x20, 0x69, 0x6e, 0x73,
						0x74, 0x61, 0x6e, 0x63, 0x65, 0x73, 0x20, 0x2a, 0x2f, 0x20, 0x20, 0x20,
						0x0d, 0x0a, 0x20, 0x20, 0x20, 0x5f, 0x5f, 0x67, 0x6c, 0x6f, 0x62, 0x61,
						0x6c, 0x20, 0x56, 0x41, 0x4c, 0x55, 0x45, 0x5f, 0x54, 0x59, 0x50, 0x45,
						0x20, 0x2a, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x2c, 0x20, 0x2f, 0x2a, 0x20,
						0x61, 0x74, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x20, 0x76, 0x65,
						0x63, 0x74, 0x6f, 0x72, 0x20, 0x2a, 0x2f, 0x0d, 0x0a, 0x09, 0x20, 0x5f,
						0x5f, 0x67, 0x6c, 0x6f, 0x62, 0x61, 0x6c, 0x20, 0x56, 0x41, 0x4c, 0x55,
						0x45, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x2a, 0x20, 0x72, 0x65, 0x73, 0x75,
						0x6c, 0x74, 0x5f, 0x6d, 0x69, 0x6e, 0x2c, 0x20, 0x2f, 0x2a, 0x20, 0x6d,
						0x69, 0x6e, 0x20, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x20, 0x70, 0x65,
						0x72, 0x20, 0x61, 0x74, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x20,
						0x2a, 0x2f, 0x0d, 0x0a, 0x09, 0x20, 0x5f, 0x5f, 0x67, 0x6c, 0x6f, 0x62,
						0x61, 0x6c, 0x20, 0x56, 0x41, 0x4c, 0x55, 0x45, 0x5f, 0x54, 0x59, 0x50,
						0x45, 0x2a, 0x20, 0x72, 0x65, 0x73, 0x75, 0x6c, 0x74, 0x5f, 0x6d, 0x61,
						0x78, 0x20, 0x2f, 0x2a, 0x20, 0x6d, 0x61, 0x78, 0x20, 0x76, 0x61, 0x6c,
						0x75, 0x65, 0x20, 0x70, 0x65, 0x72, 0x20, 0x61, 0x74, 0x74, 0x72, 0x69,
						0x62, 0x75, 0x74, 0x65, 0x20, 0x2a, 0x2f, 0x0d, 0x0a, 0x29, 0x7b, 0x0d,
						0x0a, 0x0d, 0x0a, 0x20, 0x20, 0x69, 0x6e, 0x74, 0x20, 0x6f, 0x66, 0x66,
						0x73, 0x65, 0x74, 0x20, 0x3d, 0x20, 0x67, 0x65, 0x74, 0x5f, 0x67, 0x72,
						0x6f, 0x75, 0x70, 0x5f, 0x69, 0x64, 0x28, 0x30, 0x29, 0x3b, 0x0d, 0x0a,
						0x20, 0x20, 0x69, 0x6e, 0x74, 0x20, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f,
						0x69, 0x6e, 0x64, 0x65, 0x78, 0x20, 0x3d, 0x20, 0x67, 0x65, 0x74, 0x5f,
						0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x64, 0x28, 0x30, 0x29, 0x3b,
						0x0d, 0x0a, 0x20, 0x20, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
						0x56, 0x41, 0x4c, 0x55, 0x45, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x20, 0x73,
						0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d, 0x61, 0x78, 0x5b, 0x47,
						0x52, 0x4f, 0x55, 0x50, 0x5f, 0x53, 0x49, 0x5a, 0x45, 0x5d, 0x2c, 0x20,
						0x73, 0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d, 0x69, 0x6e, 0x5b,
						0x47, 0x52, 0x4f, 0x55, 0x50, 0x5f, 0x53, 0x49, 0x5a, 0x45, 0x5d, 0x3b,
						0x0d, 0x0a, 0x20, 0x20, 0x69, 0x6e, 0x74, 0x20, 0x67, 0x78, 0x20, 0x3d,
						0x20, 0x67, 0x65, 0x74, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69,
						0x64, 0x28, 0x30, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x69, 0x6e, 0x74,
						0x20, 0x67, 0x6c, 0x6f, 0x49, 0x64, 0x20, 0x3d, 0x20, 0x67, 0x78, 0x3b,
						0x0d, 0x0a, 0x20, 0x20, 0x20, 0x0d, 0x0a, 0x20, 0x20, 0x0d, 0x0a, 0x20,
						0x20, 0x56, 0x41, 0x4c, 0x55, 0x45, 0x5f, 0x54, 0x59, 0x50, 0x45, 0x20,
						0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x6f, 0x72, 0x5f,
						0x6d, 0x69, 0x6e, 0x2c, 0x20, 0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c,
						0x61, 0x74, 0x6f, 0x72, 0x5f, 0x6d, 0x61, 0x78, 0x3b, 0x0d, 0x0a, 0x20,
						0x20, 0x69, 0x66, 0x20, 0x28, 0x67, 0x6c, 0x6f, 0x49, 0x64, 0x3c, 0x6c,
						0x65, 0x6e, 0x67, 0x74, 0x68, 0x29, 0x7b, 0x0d, 0x0a, 0x20, 0x20, 0x20,
						0x20, 0x20, 0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x6f,
						0x72, 0x5f, 0x6d, 0x61, 0x78, 0x20, 0x3d, 0x20, 0x69, 0x6e, 0x70, 0x75,
						0x74, 0x5b, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x20, 0x2b, 0x20, 0x67,
						0x78, 0x2a, 0x73, 0x74, 0x72, 0x69, 0x64, 0x65, 0x5d, 0x3b, 0x0d, 0x0a,
						0x20, 0x20, 0x20, 0x20, 0x20, 0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c,
						0x61, 0x74, 0x6f, 0x72, 0x5f, 0x6d, 0x69, 0x6e, 0x20, 0x3d, 0x20, 0x61,
						0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x6f, 0x72, 0x5f, 0x6d,
						0x61, 0x78, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x67, 0x78,
						0x20, 0x3d, 0x20, 0x67, 0x78, 0x20, 0x2b, 0x20, 0x67, 0x65, 0x74, 0x5f,
						0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x73, 0x69, 0x7a, 0x65, 0x28, 0x30,
						0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x7d, 0x0d, 0x0a, 0x20, 0x20, 0x0d,
						0x0a, 0x20, 0x20, 0x66, 0x6f, 0x72, 0x20, 0x28, 0x3b, 0x20, 0x67, 0x78,
						0x3c, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x3b, 0x20, 0x67, 0x78, 0x20,
						0x2b, 0x3d, 0x20, 0x67, 0x65, 0x74, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c,
						0x5f, 0x73, 0x69, 0x7a, 0x65, 0x28, 0x30, 0x29, 0x29, 0x7b, 0x0d, 0x0a,
						0x20, 0x20, 0x20, 0x20, 0x20, 0x56, 0x41, 0x4c, 0x55, 0x45, 0x5f, 0x54,
						0x59, 0x50, 0x45, 0x20, 0x65, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x74, 0x20,
						0x3d, 0x20, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5b, 0x6f, 0x66, 0x66, 0x73,
						0x65, 0x74, 0x20, 0x2b, 0x20, 0x67, 0x78, 0x2a, 0x73, 0x74, 0x72, 0x69,
						0x64, 0x65, 0x5d, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x69,
						0x66, 0x20, 0x28, 0x65, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x74, 0x20, 0x3c,
						0x20, 0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x6f, 0x72,
						0x5f, 0x6d, 0x69, 0x6e, 0x29, 0x20, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
						0x20, 0x7b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x09, 0x61, 0x63,
						0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x6f, 0x72, 0x5f, 0x6d, 0x69,
						0x6e, 0x20, 0x3d, 0x20, 0x65, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x74, 0x3b,
						0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x7d, 0x0d, 0x0a, 0x20, 0x20,
						0x20, 0x20, 0x20, 0x69, 0x66, 0x20, 0x28, 0x65, 0x6c, 0x65, 0x6d, 0x65,
						0x6e, 0x74, 0x20, 0x3e, 0x20, 0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c,
						0x61, 0x74, 0x6f, 0x72, 0x5f, 0x6d, 0x61, 0x78, 0x29, 0x20, 0x0d, 0x0a,
						0x20, 0x20, 0x20, 0x20, 0x20, 0x7b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20,
						0x20, 0x09, 0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x6f,
						0x72, 0x5f, 0x6d, 0x61, 0x78, 0x20, 0x3d, 0x20, 0x65, 0x6c, 0x65, 0x6d,
						0x65, 0x6e, 0x74, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x7d,
						0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x62, 0x61, 0x72, 0x72, 0x69,
						0x65, 0x72, 0x28, 0x43, 0x4c, 0x4b, 0x5f, 0x4c, 0x4f, 0x43, 0x41, 0x4c,
						0x5f, 0x4d, 0x45, 0x4d, 0x5f, 0x46, 0x45, 0x4e, 0x43, 0x45, 0x29, 0x3b,
						0x0d, 0x0a, 0x20, 0x20, 0x7d, 0x0d, 0x0a, 0x0d, 0x0a, 0x20, 0x20, 0x73,
						0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d, 0x61, 0x78, 0x5b, 0x6c,
						0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x5d, 0x20,
						0x20, 0x3d, 0x20, 0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74,
						0x6f, 0x72, 0x5f, 0x6d, 0x61, 0x78, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x73,
						0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d, 0x69, 0x6e, 0x5b, 0x6c,
						0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x5d, 0x20,
						0x20, 0x3d, 0x20, 0x61, 0x63, 0x63, 0x75, 0x6d, 0x75, 0x6c, 0x61, 0x74,
						0x6f, 0x72, 0x5f, 0x6d, 0x69, 0x6e, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x0d,
						0x0a, 0x20, 0x20, 0x62, 0x61, 0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43,
						0x4c, 0x4b, 0x5f, 0x4c, 0x4f, 0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d,
						0x5f, 0x46, 0x45, 0x4e, 0x43, 0x45, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20,
						0x69, 0x6e, 0x74, 0x20, 0x74, 0x61, 0x69, 0x6c, 0x20, 0x3d, 0x20, 0x6c,
						0x65, 0x6e, 0x67, 0x74, 0x68, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x52, 0x45,
						0x44, 0x55, 0x43, 0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d, 0x49,
						0x4e, 0x5f, 0x4d, 0x41, 0x58, 0x28, 0x20, 0x74, 0x61, 0x69, 0x6c, 0x2c,
						0x20, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78,
						0x2c, 0x20, 0x31, 0x32, 0x38, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x52,
						0x45, 0x44, 0x55, 0x43, 0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d,
						0x49, 0x4e, 0x5f, 0x4d, 0x41, 0x58, 0x28, 0x20, 0x74, 0x61, 0x69, 0x6c,
						0x2c, 0x20, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65,
						0x78, 0x2c, 0x20, 0x36, 0x34, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x52,
						0x45, 0x44, 0x55, 0x43, 0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d,
						0x49, 0x4e, 0x5f, 0x4d, 0x41, 0x58, 0x28, 0x20, 0x74, 0x61, 0x69, 0x6c,
						0x2c, 0x20, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65,
						0x78, 0x2c, 0x20, 0x33, 0x32, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x52,
						0x45, 0x44, 0x55, 0x43, 0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d,
						0x49, 0x4e, 0x5f, 0x4d, 0x41, 0x58, 0x28, 0x20, 0x74, 0x61, 0x69, 0x6c,
						0x2c, 0x20, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65,
						0x78, 0x2c, 0x20, 0x31, 0x36, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x52,
						0x45, 0x44, 0x55, 0x43, 0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d,
						0x49, 0x4e, 0x5f, 0x4d, 0x41, 0x58, 0x28, 0x20, 0x74, 0x61, 0x69, 0x6c,
						0x2c, 0x20, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65,
						0x78, 0x2c, 0x20, 0x38, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x52, 0x45,
						0x44, 0x55, 0x43, 0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d, 0x49,
						0x4e, 0x5f, 0x4d, 0x41, 0x58, 0x28, 0x20, 0x74, 0x61, 0x69, 0x6c, 0x2c,
						0x20, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78,
						0x2c, 0x20, 0x34, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x52, 0x45, 0x44,
						0x55, 0x43, 0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d, 0x49, 0x4e,
						0x5f, 0x4d, 0x41, 0x58, 0x28, 0x20, 0x74, 0x61, 0x69, 0x6c, 0x2c, 0x20,
						0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x2c,
						0x20, 0x32, 0x29, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x52, 0x45, 0x44, 0x55,
						0x43, 0x45, 0x5f, 0x53, 0x54, 0x45, 0x50, 0x5f, 0x4d, 0x49, 0x4e, 0x5f,
						0x4d, 0x41, 0x58, 0x28, 0x20, 0x74, 0x61, 0x69, 0x6c, 0x2c, 0x20, 0x6c,
						0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78, 0x2c, 0x20,
						0x31, 0x29, 0x3b, 0x0d, 0x0a, 0x0d, 0x0a, 0x20, 0x20, 0x69, 0x66, 0x20,
						0x28, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x64, 0x65, 0x78,
						0x3d, 0x3d, 0x30, 0x29, 0x7b, 0x0d, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x20,
						0x72, 0x65, 0x73, 0x75, 0x6c, 0x74, 0x5f, 0x6d, 0x61, 0x78, 0x5b, 0x20,
						0x67, 0x65, 0x74, 0x5f, 0x67, 0x72, 0x6f, 0x75, 0x70, 0x5f, 0x69, 0x64,
						0x28, 0x30, 0x29, 0x5d, 0x20, 0x20, 0x3d, 0x20, 0x73, 0x63, 0x72, 0x61,
						0x74, 0x63, 0x68, 0x5f, 0x6d, 0x61, 0x78, 0x5b, 0x30, 0x5d, 0x3b, 0x0d,
						0x0a, 0x20, 0x20, 0x20, 0x20, 0x20, 0x72, 0x65, 0x73, 0x75, 0x6c, 0x74,
						0x5f, 0x6d, 0x69, 0x6e, 0x5b, 0x20, 0x67, 0x65, 0x74, 0x5f, 0x67, 0x72,
						0x6f, 0x75, 0x70, 0x5f, 0x69, 0x64, 0x28, 0x30, 0x29, 0x5d, 0x20, 0x20,
						0x3d, 0x20, 0x73, 0x63, 0x72, 0x61, 0x74, 0x63, 0x68, 0x5f, 0x6d, 0x69,
						0x6e, 0x5b, 0x30, 0x5d, 0x3b, 0x0d, 0x0a, 0x20, 0x20, 0x7d, 0x0d, 0x0a,
						0x20, 0x20, 0x0d, 0x0a, 0x20, 0x20, 0x72, 0x65, 0x74, 0x75, 0x72, 0x6e,
						0x3b, 0x0d, 0x0a, 0x7d,0x0
					};
					unsigned int min_max_cl_len = 3304;


				



					code.append((const char*)min_max_cl);
					ctx.add_program(code, program_name());

				}


			
			};
		}
	}
}


#endif 
