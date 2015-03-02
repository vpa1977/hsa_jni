package org.moa.streams;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;

import org.stream_gpu.knn.kdtree.KDTreeWindow;

import moa.core.Measurement;
import weka.core.Instance;


public class ParallelKdTree extends AbstractParallelClassifier {
	
	private KDTreeWindow m_window;
	private LinkedBlockingQueue<Prediction> m_queue;
	private int m_k;
	
	public ParallelKdTree(KDTreeWindow window, int k)
	{
		m_window = window;
		m_queue = new LinkedBlockingQueue<Prediction>();
		m_window.setWorkQueue(m_queue);
		m_k = k;
	}

	public void synchronizeTraining() {
		
	}

	public void evaluate(Prediction prediction) {
		m_window.evaluate(m_k, prediction);		
	}

	public Prediction take() {
		if (m_window.hasPendingQueries())
		{
			try {
				return m_queue.take();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return null;
	}

	public Prediction poll() {
		return m_queue.poll();
	}

	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void resetLearningImpl() {
		
		m_window.clear();
		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		m_window.add(inst);
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}

}
