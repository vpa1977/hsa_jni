package org.moa.streams;


import moa.classifiers.AbstractClassifier;
import moa.core.InstancesHeader;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;
import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 * Wrapper 
 * @author bsp
 *
 */
public abstract class AbstractParallelClassifier extends AbstractClassifier implements ParallelClassifier {

	public double[] getVotesForInstance(Instance inst) {
		evaluate(new Prediction(inst));
		return take().getVotes();
	}

}
