package weka.core.neighboursearch;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;

public interface IProjection {
	
	Instance[] addToProjections(Instance src) throws Exception;

	/** 
	 * project compatible dataset 
	 * @param dataset
	 * @return
	 * @throws Exception 
	 */
	Instances[] project(Instances dataset) throws Exception;

	/** 
	 * 
	 * @return current projection result
	 */
	Instances[] getProjection();

	/** 
	 * create a projection
	 * @param curve
	 * @param src
	 * @return
	 * @throws Exception
	 */
	Instance project(int curve, Instance src) throws Exception;

	double[][] getRanges(int i);

	NormalizableDistance getDistance(int i);

}