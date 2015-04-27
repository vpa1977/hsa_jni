package org.moa.gpu;

/** 
 * Context provides interface to the viennacl-backed ML algorithms implementations
 * @author bsp
 *
 */
public class Context {
	static
	{
		System.loadLibrary("moa-frontend-lib");
	}
	


	public static void load() {
		// dummy method to load native library
	}
}
