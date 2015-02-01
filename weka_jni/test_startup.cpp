#include "context.hpp"
#include <sys/time.h>

#define BILLION 1000000000L

void init_sample(double* ptr, double* sample, int instance_size, int window_size, int offset)
{
	for (int i = 0 ;i < instance_size ; ++i)
		ptr[i*window_size+offset] = sample[i];
}

void find_min_max(double* samples, int window_size, int offset)
{

}

void fill_values_indices(double* ptr, int* ind, int size)
{
	for (int i = 0 ; i < size ; ++i)
	{
		ptr[i] = 1;
		ind[i] = i;
	}
}

void testProduct()
{
	Algorithms test(std::shared_ptr<HSAContext>(HSAContext::Create()));
	#define TEST_SIZE 127
	#define INST_SIZE 511
		double* values[TEST_SIZE];
		int* indices[TEST_SIZE];
		int size2d[TEST_SIZE];
		for (int i = 0 ; i < TEST_SIZE; ++i)
		{
			values[i] = new double[INST_SIZE];
			indices[i] = new int[INST_SIZE];
			fill_values_indices(values[i],indices[i], INST_SIZE);
			size2d[i] = INST_SIZE;
		}
		double weights[INST_SIZE];
		for (int i = 0 ; i < INST_SIZE; ++i)
			weights[i] = 1;
		double temp[TEST_SIZE* 384];
		memset(temp, 0 , sizeof(temp));
		//memset(temp, 0, sizeof(double)* 320 * 2);
		test.m_sparse_product_2d.product(TEST_SIZE, 0, (double**)values, (int**)indices, (double*)weights,(int*)size2d, temp );

		printf("product result %f, %f, %f\n\n", temp[0], temp[1], temp[2]);
}

// return diff in microseconds
long time_difference(timeval tv1, timeval tv2)
{
	return ((tv2.tv_sec - tv1.tv_sec)*1000000L    +tv2.tv_usec) - tv1.tv_usec;
}

void testSum()
{
	int workgroup_size = 256;
	int compute_nodes = 1;
	Algorithms test(std::shared_ptr<HSAContext>(HSAContext::Create()));

	timeval tv1;
	timeval tv2;

	FILE* output = fopen("/tmp/log.txt", "w+");

	std::vector<double> input1;
	std::vector<double> input2;
	std::vector<double> result;
	for (int global_size = 1024; global_size <= 1024 ; global_size += 256)
	for (compute_nodes = 1 ; compute_nodes <= global_size/256; ++compute_nodes)
	{
		int size = workgroup_size * compute_nodes;
		input1.resize(size);
		input2.resize(size);
		result.resize(size);
		memset(&result[0], 0, size * sizeof(double));
		gettimeofday(&tv1, 0);
		test.m_vector_sum.begin();
		for (int i = 0 ;i < 100 ; ++i)
		{
			test.m_vector_sum.compute(global_size,&input1[0],&input2[0], &result[0], size, i==99);

		}
		test.m_vector_sum.commit();
		gettimeofday(&tv2, 0);
		for (int i = 0 ;i < size-1; ++i)
			if (result[i] != 1000)
				printf("invalid code\n");
		fprintf(output,"%d,%d,%ld\n",global_size,  compute_nodes,  time_difference(tv1,tv2)/100);
		//printf("Size %d Warps scheduled %d Runtime is %ld \n", size, compute_nodes,  time_difference(tv1,tv2)/10000);
	}
	fclose(output);
}

void testSumTemplateDispatch()
{
	int workgroup_size = 256;
		int compute_nodes = 1;
		Algorithms test(std::shared_ptr<HSAContext>(HSAContext::Create()));

		timeval tv1;
		timeval tv2;

		FILE* output = fopen("/tmp/log.txt", "w+");

		std::vector<double> input1;
		std::vector<double> input2;
		std::vector<double> result;
		for (int global_size = 1024; global_size <= 1024 ; global_size += 256)
		for (compute_nodes = 1 ; compute_nodes <= global_size/256; ++compute_nodes)
		{
			int size = workgroup_size * compute_nodes;
			input1.resize(size);
			input2.resize(size);
			result.resize(size);
			gettimeofday(&tv1, 0);

			for (int i = 0 ;i < 100000 ; ++i)
			{
				test.m_vector_sum_template.compute(global_size,&input1[0],&input2[0], &result[0], size);
			}

			gettimeofday(&tv2, 0);
			if (result[size-1] != size-1)
				printf("invalid code\n");
			fprintf(output,"%d,%d,%ld\n",global_size,  compute_nodes,  time_difference(tv1,tv2)/100000);
			//printf("Size %d Warps scheduled %d Runtime is %ld \n", size, compute_nodes,  time_difference(tv1,tv2)/10000);
		}
		fclose(output);
}

void testDistance()
{
	Algorithms test(std::shared_ptr<HSAContext>(HSAContext::Create()));
	int element_size = 256;
	int window_size = 1024*128;
	int numerics_size = 512;

	std::vector<double> query;
	std::vector<double> window;
	std::vector<double> ranges;
	std::vector<double> result;
	query.resize(element_size);
	window.resize(element_size * window_size);
	result.resize(window_size);

	ranges.resize(element_size*2);
	for (int i = 0 ;i < element_size ; ++i)
	{
		ranges[2*i] = 1;
		ranges[2*i+1] = 2;
	}

	timeval tv1;
	timeval tv2;

	gettimeofday(&tv1, 0);
	for (int i = 0 ;i < 1000 ; ++i)
		test.m_square_distance2.calculate(&query[0], &window[0], window_size, &ranges[0], element_size, numerics_size, &result[0]);
	gettimeofday(&tv2, 0);
	printf("%ld\n", time_difference(tv1,tv2)/1000);

}

int main()
{
	printf("test start\n");
	testSum();

	/*	int instance_size = 16;
	//int numeric_size = 10;
	int window_size = 1;

	double test_vector[16];
	double test_sample[16 * 256];
	//double distances[256];
	//double ranges[32];
	for (int i = 0 ; i < 16 ; ++i)
		test_vector[i] = 1;
	for (int i = 0 ;i < window_size ; ++i )
		init_sample(test_sample, test_vector, instance_size, window_size, i);


	std::vector<double> reduce_result = test.m_reduce.reduce(test_sample, 16, 256);

	if (reduce_result.size() != 256)
		printf("wtf!!!");
*/

}
