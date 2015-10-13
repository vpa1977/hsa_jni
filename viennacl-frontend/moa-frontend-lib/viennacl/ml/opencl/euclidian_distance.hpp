#ifndef MIN_MAX_HPP
#define MIN_MAX_HPP

#include <viennacl/ocl/context.hpp>

namespace viennacl
{
	namespace ml
	{
		namespace opencl
		{
			template <typename NumericT, class context_class>
			struct euclidian_distance_kernel
			{

				static std::string program_name()
				{
					return "knn";
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
					
					/* from euclidian_distance.cl*/


				



					code.append((const char*)min_max_cl);
					ctx.add_program(code, program_name());

				}


			
			};
		}
	}
}


#endif 
