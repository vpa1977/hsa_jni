package org.moa.gpu;

import weka.core.Instance;

public interface Window {

	/* 
	 * Clear the window
	 */
	public void clear();
	/** 
	 * Add instance to the window (equal time intervals)
	 * @param inst
	 */
	public void add(Instance inst);
	/** 
	 * Add instance with arbitrary time
	 * @param inst
	 * @param t
	 */
	public void add(Instance inst, long t);
	/** 
	 * get instances
	 * @return
	 */
	public Instance[] get();
	
	/** 
	 * get attribute values 
	 * @return
	 */
	public double[] getValues();
	public int[] getRowIndices();
	public int[] getColumnIndices();
	public double[] getClassValues();
	public int getRowCount();
	public int getColumnCount();
	
	public boolean full();

}
