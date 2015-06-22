/*
 * knn_sliding_window.hpp
 *
 *  Created on: 2/06/2015
 *      Author: bsp
 */

#ifndef KNN_SLIDING_WINDOW_HPP_
#define KNN_SLIDING_WINDOW_HPP_

#include <viennacl/vector.hpp>

/**
 * This is a backing for the dense sliding window.
 */
class dense_sliding_window
{
public:
	dense_sliding_window() : m_row_index(0) {}
private:
	viennacl::vector<int> m_attribute_types;
	viennacl::matrix<double> m_values_window;
	viennacl::vector<double> m_classes_values;
	int m_row_index;
};


viennacl::vector<double> distance(const dense_sliding_window& sliding_window, int start_row, int end_row, const viennacl::vector<double>& sample);



#endif /* KNN_SLIDING_WINDOW_HPP_ */
