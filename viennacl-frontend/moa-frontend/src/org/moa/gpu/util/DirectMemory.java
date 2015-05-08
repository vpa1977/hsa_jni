package org.moa.gpu.util;

import java.lang.reflect.Constructor;

import sun.misc.Unsafe;

public class DirectMemory {
	
	
	public static long allocate(long length)
	{
		long handle =m_direct_memory.allocateMemory(length);
		return handle;
	}
	
	public static void set(long handle, long size, byte value)
	{
		m_direct_memory.setMemory(handle, size, value);
	}
	
	public static void free(long handle)
	{
		m_direct_memory.freeMemory(handle);
		
	}
	
	public static void write(long handle, int index, double value)
	{
		m_direct_memory.putDouble(handle +index(DOUBLE_SIZE, index), value);
	}
	
	public static long read(long handle, int index)
	{
		return m_direct_memory.getLong(handle + index(LONG_SIZE, index));
	}
	
	public static void write(long handle, int index, long value)
	{
		m_direct_memory.putLong(handle + index(LONG_SIZE, index), value);
	}
	
	public static void write(long handle, int index, int value)
	{
		m_direct_memory.putLong(handle + index(INT_SIZE, index), value);
	}

	
	private static long index(long size_in_bytes, int index)
	{
		return size_in_bytes * index;
	}
	
	public static long DOUBLE_SIZE = 8; // todo fix in native code.
	public static long INT_SIZE = 4; // todo fix in native code.
	public static final long LONG_SIZE = 8; // todo fix in native code
	private static Unsafe m_direct_memory = getUnsafe();
	private static Unsafe getUnsafe()
	{
		try {
			Constructor<Unsafe> unsafeConstructor = Unsafe.class.getDeclaredConstructor();
			unsafeConstructor.setAccessible(true);
			Unsafe unsafe = unsafeConstructor.newInstance();
			return unsafe;
		}
		catch (Throwable t){}
		return null;
	}

}
