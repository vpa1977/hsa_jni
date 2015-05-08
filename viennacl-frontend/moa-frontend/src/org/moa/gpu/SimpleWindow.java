package org.moa.gpu;

import java.lang.reflect.Constructor;
import sun.misc.Unsafe;
import java.util.ArrayList;

import weka.core.Instance;

public class SimpleWindow implements Window {
	
	private Instance[] m_instances;
	
	private int[] m_rows;
	private int[] m_columns;
	private double[] m_values;
	private double[] m_classes;
	
	private int m_index;
	private boolean m_full;

	public SimpleWindow(int size)
	{
		m_instances = new Instance[size];
		clear();
	}
	

	@Override
	public void clear() {
		for (int i= 0 ; i < m_instances.length; ++i)
			m_instances[i] = null;
		m_index = 0;
		m_full = false;
		m_rows = null;
		m_columns = null;
		m_values = null;
		m_classes = null;
	}

	@Override
	public void add(Instance inst) {
		m_instances[m_index] = inst;
		++m_index;
		if (m_index >= m_instances.length)
		{
			m_index = 0;
			m_full = true;
		}
		
	}

	@Override
	public Instance[] get() {
		return m_instances;
	}

	@Override
	public void add(Instance inst, long t) {
		add(inst);
		
	}

	@Override
	public boolean full() {
		return m_full;
	}

	@Override
	public double[] getValues() {
		initArrays();
		return m_values;
	}

	@Override
	public int[] getRowIndices() {
		initArrays();
		return m_rows;
	}

	@Override
	public int[] getColumnIndices() {
		initArrays();
		return m_columns;
	}

	@Override
	public double[] getClassValues() {
		initArrays();
		return m_classes;
	}

	private void initArrays() {
		if (m_rows != null)
			return;
		ArrayList<Integer> row_list = new ArrayList<Integer>();
		ArrayList<Integer> column_list = new ArrayList<Integer>();
		ArrayList<Double> value_list = new ArrayList<Double>();
		int classIndex = m_instances[0].classIndex();
		int numAttributes = m_instances[0].numAttributes();
		m_classes = new double[ m_instances.length];
		for (int row = 0 ; row < m_instances.length ; ++row)
		{
			Instance next  = m_instances[row];
			for (int i = 0 ; i < numAttributes ; ++i)
			{
				int column = i < classIndex ? i  : i -1;
				if (!next.isMissing(i) && i != classIndex)
				{
					row_list.add(row);
					column_list.add( column );
					value_list.add(next.value(i));
				}
			}
			m_classes[row] = next.value(classIndex);
		}
		
		m_rows = new int[row_list.size()];
		m_columns = new int[column_list.size()];
		m_values = new double[ value_list.size()];
		
		for (int i = 0 ;i < m_rows.length ; ++i)
		{
			m_rows[i] = row_list.get(i);
			m_columns[i] = column_list.get(i);
			m_values[i] = value_list.get(i);
		}
	}

	@Override
	public int getRowCount() {
		return m_instances.length;
	}

	@Override
	public int getColumnCount() {
		return m_instances[0].numAttributes() -1;
	}
	
	

}
