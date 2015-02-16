package org.moa.streams;

import moa.core.InstancesHeader;
import moa.core.Measurement;
import weka.core.Instance;

public interface ParallelClassifier {

	/** 
	 * schedules training of the classifier on the provided instance
	 * @param inst
	 */
	void trainOnInstance(Instance inst);
	/** 
	 * Completes training queue
	 */
	void synchronizeTraining();
	
	void evaluate(Prediction prediction);
	void setModelContext(InstancesHeader context);
	Measurement[] getModelMeasurements();
	
	/** 
	 * Block until prediction is ready or return null if no more pending predictions
	 * @return
	 */
	Prediction take();

	/**
	 * return complete prediction or null if none is ready
	 * @return
	 */
	Prediction poll();

	
}
