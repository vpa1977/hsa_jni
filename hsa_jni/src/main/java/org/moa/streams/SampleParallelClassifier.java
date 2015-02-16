package org.moa.streams;

import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;
import moa.classifiers.Classifier;
import weka.core.Instance;

public class SampleParallelClassifier extends AbstractParallelClassifier {
	
	private Classifier m_classifier;
	private PriorityBlockingQueue<Prediction> m_results = new PriorityBlockingQueue<Prediction>();
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train", 
			Classifier.class, "moa.classifiers.bayes.NaiveBayes");

	private ThreadPoolExecutor m_executor;
	
	private Semaphore m_semaphore = new Semaphore(0);
	private AtomicInteger m_task_count = new AtomicInteger(0);
	
	private class PredictionTask implements Runnable
	{
		public Prediction m_prediction;
		
		public PredictionTask(Prediction p)
		{
			m_prediction = p;
		}

		public void run() {
			double[] votes = m_classifier.getVotesForInstance(m_prediction.instance());
			m_prediction.setVotes(votes);
			m_results.add(m_prediction);
			m_task_count.decrementAndGet();
		}
		
	}
			

	public void synchronizeTraining() {
		// do nothing
	}

	public void evaluate(final Prediction prediction) {
		 m_executor.submit(new PredictionTask(prediction));
		 m_task_count.incrementAndGet();
	}

	public Prediction take() {
		Prediction result = m_results.poll();
		if (m_task_count.intValue() == 0)
			return result;
		try {
			return m_results.take();
		}
		catch (InterruptedException e){
		}
		return null;
	}

	public Prediction poll() {
		return m_results.poll();
	}

	public boolean isRandomizable() {
		return true;
	}

	@Override
	public void resetLearningImpl() {
		m_classifier.resetLearning();
		m_executor = new ThreadPoolExecutor(10, 100, 1000, TimeUnit.SECONDS, new PriorityBlockingQueue<Runnable>());
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		m_classifier.trainOnInstance(inst);
	}

	@Override
	public void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
		// TODO Auto-generated method stub
		super.prepareForUseImpl(monitor, repository);
		m_classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
	 
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return m_classifier.getModelMeasurements();
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		
		
	}

}
