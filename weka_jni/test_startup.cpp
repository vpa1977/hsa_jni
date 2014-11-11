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
		test_vector[i] = i;
	for (int i = 0 ;i < window_size ; ++i )
		init_sample(test_sample, test_vector, instance_size, window_size, i);

	for (int i = 0 ; i < numeric_size; ++i)
	{
		int offset = i*window_size;
		int off = i*2;
		test.m_min_max_value.find(test_sample+offset,window_size,ranges[off], ranges[off+1] );
	}

	for (int i = 0 ;i < 16; i++)
	{
		++test_vector[i];
		ranges[i*2+1]++;
	}



	test.m_per_attribute_distances.distance(test_vector,
			test_sample, ranges,
			numeric_size, instance_size - numeric_size,
			window_size, window_size, distances);
for (int i = 0 ;  i < window_size; ++i)
	printf(" %f \n", distances[i]);
/*	Algorithms test(std::shared_ptr<HSAContext>(HSAContext::Create()));


	double arg[64*6*256];
	arg[30] = 2222;
	for (int i = 0 ; i < 10; ++i)
		test.m_dump.run(256,arg);

	printf("Measuring startup for dump\n");

	int iterations = 10000;
	timespec ts, ts1;
	long clk = clock();
	clock_gettime(CLOCK_MONOTONIC, &ts);
	double max = 0;
	for (int i = 0 ; i < iterations ; ++i)
		max = test.m_max_value.find(arg, 64*6*256, 1, 0);
	clock_gettime(CLOCK_MONOTONIC, &ts1);
	uint64_t	 diff = BILLION * (ts1.tv_sec - ts.tv_sec) + ts1.tv_nsec - ts.tv_nsec;
	double val = ((double)diff/1000000); // msec
	val = val / iterations;
	long clc2 = clock();
	printf("HSA elapsed time = %f msec\n", val);
	//printf("maxs %f, \n", max);
	printf("test %f\n", (double)(clc2 - clk)/(CLOCKS_PER_SEC*iterations));
	*/

}
