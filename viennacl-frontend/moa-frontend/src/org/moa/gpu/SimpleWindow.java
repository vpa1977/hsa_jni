package org.moa.gpu;

import weka.core.Instance;

public class SimpleWindow implements Window {
	
	private Instance[] m_instances;
	private int m_index;
	private boolean m_full;

	public SimpleWindow(int size)
	{
		m_instances = new Instance[size];
		m_index = 0;
		m_full = false;
	}

	@Override
	public void clear() {
		for (int i= 0 ; i < m_instances.length; ++i)
			m_instances[i] = null;
		m_index = 0;
		m_full = false;
		
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

}
