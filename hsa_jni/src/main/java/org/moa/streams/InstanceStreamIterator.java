package org.moa.streams;

import java.util.Iterator;

import moa.streams.InstanceStream;

import org.reactivestreams.Publisher;
import org.reactivestreams.Subscriber;

import weka.core.Instance;
import akka.stream.javadsl.Source;


/** 
 * Iterator wrapper over instance stream   
 * @author bsp
 *
 */
public class InstanceStreamIterator implements Iterator<Instance> {
	
	private InstanceStream m_instance_stream;
	
	public InstanceStreamIterator(InstanceStream it)
	{
		m_instance_stream = it;
	}

	public boolean hasNext() {
		return m_instance_stream.hasMoreInstances();
	}

	public Instance next() {
		return m_instance_stream.nextInstance();
	}

}
