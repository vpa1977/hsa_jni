#include "context.hpp"

int main()
{
	printf("test start\n");
	Algorithms test(HSAContext::Create());

	double* vect = new double[64];
	for (int i = 0 ;i < 64; ++ i )
		vect[i] = 10;
	vect[0] = 22;
	vect[1 + 2*4] = 33;
	int * indices = new int[64];

	double max = test.m_max_value.find( vect, 16, 4, 0);
	double max1 = test.m_max_value.find( vect, 16, 4, 1);

	printf("Must be 22 %f\n",max1);


}
