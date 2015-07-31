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
	
	public static void write(long handle, double value)
	{
		m_direct_memory.putDouble(handle, value);
	}
	
	public static double read(long handle)
	{
		return m_direct_memory.getDouble(handle);
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
