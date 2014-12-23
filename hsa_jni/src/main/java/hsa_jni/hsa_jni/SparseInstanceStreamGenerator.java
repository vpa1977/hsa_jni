package hsa_jni.hsa_jni;

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
		gen.numNumericsOption.setValue(1024);
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

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		m_generator.getDescription(sb, indent);
	}

	@Override
	public InstancesHeader getHeader() {
		return m_generator.getHeader();
	}

	@Override
	public long estimatedRemainingInstances() {
		return m_generator.estimatedRemainingInstances();
	}

	@Override
	public boolean hasMoreInstances() {
		return m_generator.hasMoreInstances();
	}

	@Override
	public Instance nextInstance() {
		return new SparseInstanceAccess( m_generator.nextInstance());
	}

	@Override
	public boolean isRestartable() {
		return m_generator.isRestartable();
	}

	@Override
	public void restart() {
		m_generator.restart();
	}


	@Override
	public void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
		((OptionHandler)m_generator).prepareForUse(monitor, repository);
		
	}

}
