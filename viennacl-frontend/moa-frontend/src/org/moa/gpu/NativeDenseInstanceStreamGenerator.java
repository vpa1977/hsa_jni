package org.moa.gpu;

import org.moa.gpu.bridge.NativeDenseInstance;

import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import weka.core.DenseInstance;
import weka.core.Instance;

public class NativeDenseInstanceStreamGenerator extends AbstractOptionHandler implements InstanceStream {
	
	public ClassOption baseStreamOption = new ClassOption("baseStream", 'b',
            "Base stream to produce instances.", InstanceStream.class, "moa.streams.generators.RandomTreeGenerator");
	
	private InstanceStream m_generator;
	private InstancesHeader m_header;
	
	public NativeDenseInstanceStreamGenerator()
	{
		throw new RuntimeException("This class is obsolete. Do not use");
	}
		

	@Override
	public int measureByteSize() {
		// TODO Auto-generated method stub
		return m_generator.measureByteSize();
	}

	
	
	public void getDescription(StringBuilder sb, int indent) {
		m_generator.getDescription(sb, indent);
	}

	
	public InstancesHeader getHeader() {
		return m_header;
	}

	
	public long estimatedRemainingInstances() {
		return m_generator.estimatedRemainingInstances();
	}

	
	public boolean hasMoreInstances() {
		return m_generator.hasMoreInstances();
	}

	
	public Instance nextInstance() {
		Instance inst = m_generator.nextInstance();
		inst.setDataset(getHeader());
		NativeDenseInstance nativeInstance = new NativeDenseInstance((DenseInstance)inst);
		return nativeInstance;
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
		m_generator = (InstanceStream)getPreparedClassOption(baseStreamOption);
		m_header = m_generator.getHeader();
	}


	
	
	

}
