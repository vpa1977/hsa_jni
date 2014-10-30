/*
 * context.hpp
 *
 *  Created on: 19/10/2014
 *      Author: bsp
 */

#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_


#include <jni.h>
#include "hsa_jni_hsa_jni_WekaHSAContext.h"
#include "hsa_jni_hsa_jni_WekaHSAContext_KnnNativeContext.h"
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <cmath>
#include "HSAContext.h"
// algorithms
#include "max_value.hpp"
#include "merge_sort.hpp"
#include "distance.hpp"

struct Algorithms
{
	Algorithms(HSAContext* p_context) : m_pcontext(p_context),
				   m_max_value(m_pcontext, "/home/bsp/hsa_jni/kernels/max_value.brig"),
				   m_min_value(m_pcontext, "/home/bsp/hsa_jni/kernels/min_value.brig"),
				   m_merge_sort(m_pcontext,
						   "/home/bsp/hsa_jni/kernels/local_merge.brig",
						   "/home/bsp/hsa_jni/kernels/global_merge.brig"),
			       m_square_distance(m_pcontext, "/home/bsp/hsa_jni/kernels/distance.brig")
{
}
	~Algorithms()
	{
		// ugly/refactor asap
		m_max_value.~MaxValue();
		m_min_value.~MinValue();
		m_merge_sort.~MergeSort();
		m_square_distance.~SquareDistance();

		delete m_pcontext;
	}
	HSAContext* m_pcontext;
	MaxValue m_max_value;
	MinValue m_min_value;
	MergeSort m_merge_sort;
	SquareDistance m_square_distance;
};

#endif /* CONTEXT_HPP_ */
