package org.moa.gpu;

import weka.core.Instance;
import moa.MOAObject;
import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.OptionHandler;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.TaskMonitor;

public class SparseInstanceStreamGenerator extends AbstractOptionHandler implements InstanceStream {
	
	private InstanceStream m_generator;
	
	public SparseInstanceStreamGenerator()
	{
		RandomTreeGenerator gen = new RandomTreeGenerator();
		gen.numNumericsOption.setValue(2048);
		m_generator = gen;
	
	}
	
	public SparseInstanceStreamGenerator(InstanceStream is)
	{
		m_generator = is;
	}
	
	

	@Override
	public int measureByteSize() {
		// TODO Auto-generated method stub
		return m_generator.measureByteSize();
	}

	public OptionHandler copy() {
		return new SparseInstanceStreamGenerator(m_generator);
	}

	
	public void getDescription(StringBuilder sb, int indent) {
		m_generator.getDescription(sb, indent);
	}

	
	public InstancesHeader getHeader() {
		return m_generator.getHeader();
	}

	
	public long estimatedRemainingInstances() {
		return m_generator.estimatedRemainingInstances();
	}

	
	public boolean hasMoreInstances() {
		return m_generator.hasMoreInstances();
	}

	
	public Instance nextInstance() {
		return new AccessibleSparseInstance( m_generator.nextInstance());
	}


	public boolean isRestartable() {
		return m_generator.isRestartable();
	}


	public void restart() {
		m_generator.restart();
	}


	@Override
	public void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
		((OptionHandler)m_generator).prepareForUse(monitor, repository);
		
	}

}
