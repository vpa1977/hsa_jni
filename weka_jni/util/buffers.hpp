/*
 * buffers.hpp
 *
 *  Created on: 27/10/2014
 *      Author: bsp
 */

#ifndef UTIL_BUFFERS_HPP_
#define UTIL_BUFFERS_HPP_

template <
	int BUFFER_COUNT,
	typename _KeyType,
	typename _ValueType = util::NullType>
struct MultiBuffer
{
	typedef _KeyType	KeyType;
	typedef _ValueType 	ValueType;

	// Set of device vector pointers for keys
	KeyType* d_keys[BUFFER_COUNT];

	// Set of device vector pointers for values
	ValueType* d_values[BUFFER_COUNT];

	// Selector into the set of device vector pointers (i.e., where the results are)
	int selector;

	// Constructor
	MultiBuffer()
	{
		selector = 0;
		for (int i = 0; i < BUFFER_COUNT; i++) {
			d_keys[i] = NULL;
			d_values[i] = NULL;
		}
	}
};

/**
 * Double buffer (a.k.a. page-flip, ping-pong, etc.) version of the
 * multi-buffer storage abstraction above.
 *
 * Many of the B40C primitives are templated upon the DoubleBuffer type: they
 * are compiled differently depending upon whether the declared type contains
 * keys-only versus key-value pairs (i.e., whether ValueType is util::NullType
 * or some real type).
 *
 * Declaring keys-only storage wrapper:
 *
 * 		DoubleBuffer<KeyType> key_storage;
 *
 * Declaring key-value storage wrapper:
 *
 * 		DoubleBuffer<KeyType, ValueType> key_value_storage;
 *
 */
template <
	typename KeyType,
	typename ValueType = util::NullType>
struct DoubleBuffer : MultiBuffer<2, KeyType, ValueType>
{
	typedef MultiBuffer<2, KeyType, ValueType> ParentType;

	// Constructor
	DoubleBuffer() : ParentType() {}

};



#endif /* UTIL_BUFFERS_HPP_ */
