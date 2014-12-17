#include "context.hpp"

#define BILLION 1000000000L

void init_sample(double* ptr, double* sample, int instance_size, int window_size, int offset)
{
	for (int i = 0 ;i < instance_size ; ++i)
		ptr[i*window_size+offset] = sample[i];
}

void find_min_max(double* samples, int window_size, int offset)
{

}

int main()
{
	printf("test start\n");

	Algorithms test(std::shared_ptr<HSAContext>(HSAContext::Create()));




	int instance_size = 16;
	int numeric_size = 10;
	int window_size = 1;

	double test_vector[16];
	double test_sample[16 * 256];
	double distances[256];
	double ranges[32];
	for (int i = 0 ; i < 16 ; ++i)
		test_vector[i] = 1;
	for (int i = 0 ;i < window_size ; ++i )
		init_sample(test_sample, test_vector, instance_size, window_size, i);


	std::vector<double> reduce_result = test.m_reduce.reduce(test_sample, 16, 256);

	if (reduce_result.size() != 256)
		printf("wtf!!!");


}
