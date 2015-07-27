package org.moa.gpu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;

import org.moa.gpu.bridge.NativeSparseInstance;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import moa.MOAObject;
import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.options.IntOption;
import moa.options.OptionHandler;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.TaskMonitor;

public class NativeSparseInstanceStreamGenerator extends AbstractOptionHandler implements InstanceStream {
	
	public ClassOption baseStreamOption = new ClassOption("baseStream", 'b',
            "Base stream to produce instances.", InstanceStream.class, "moa.streams.generators.RandomTreeGenerator");
	
	public IntOption numAttributesOption = new IntOption("numAttributes", 'a', "Number of attributes ", 1024);
	public IntOption numMappingsOption   = new IntOption("numMappings", 'm', "Number of mappings", 10);
	
	
	private InstanceStream m_generator;
	private ArrayList< HashMap<Integer,Integer> > m_mapping;
	private Random m_random = new Random();

	private InstancesHeader m_header;
	
	public NativeSparseInstanceStreamGenerator()
	{
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
		AccessibleSparseInstance ac =  new AccessibleSparseInstance(m_header, m_generator.nextInstance(), m_mapping.get( m_random.nextInt(m_mapping.size())));
		NativeSparseInstance nativeInstance = new NativeSparseInstance(ac); 
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
		InstancesHeader base_header = m_generator.getHeader();
		int base_num_attributes = m_generator.getHeader().numAttributes();
		int num_attributes = numAttributesOption.getValue();
		int num_mappings = numMappingsOption.getValue();
		int class_index = base_header.classIndex();
		ArrayList<Integer> bag = new ArrayList<Integer>();
		ArrayList<weka.core.Attribute> info = new ArrayList<weka.core.Attribute>();
		for (int i = 0; i < num_attributes; ++i)
		{
			info.add( new Attribute(""+i));
			if (i != class_index)
				bag.add(i);
		}
		
		m_mapping = new ArrayList< HashMap<Integer,Integer>> ();
		Random rnd = new Random();
		for (int mapping = 0; mapping < num_mappings ; ++mapping)
		{
			HashMap<Integer,Integer> attr_map = new HashMap<Integer,Integer>();
			for (int i = 0; i < base_num_attributes ; ++i)
			{
				
				if (i == class_index)
					continue;
				int attribute = i;
				if (bag.size() == 0)
					throw new RuntimeException("Not enough attributes to create a new mapping");
				int mapped_index = 0;
				while ((mapped_index=rnd.nextInt(bag.size())) == class_index)
				{} // do not map to the class index
				
				int mapped = bag.remove(mapped_index);
				Attribute src = (Attribute)base_header.attribute(i).copy(base_header.attribute(i).name() +  "_to_"+mapped);
				info.set(mapped,  src);				
				attr_map.put(attribute,mapped);
			}
			
			attr_map.put(class_index,class_index);
			info.set(class_index, (Attribute)base_header.attribute(class_index).copy());
		//	printAttributeMap(attr_map, class_index);
			m_mapping.add( attr_map );
		}
		
		Instances sparseHeader = new Instances("sparse_"+base_header.relationName(), info, 1);
		sparseHeader.setClassIndex(class_index);
		m_header = new InstancesHeader(sparseHeader);
	}


	private void printAttributeMap(HashMap<Integer, Integer> attr_map,
			int class_index) {
		Iterator<Entry<Integer,Integer>> contents = attr_map.entrySet().iterator();
		System.out.println("*");
		while (contents.hasNext())
		{
			Entry<Integer,Integer> entry = contents.next();
			String dump = entry.getKey() + " => " +entry.getValue() + " " + ( class_index == entry.getKey() ? "*" : " ");
			System.out.println(dump);
		}
		System.out.println("*");
		
		
	}
	
	

}
