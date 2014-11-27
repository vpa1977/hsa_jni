package hsa_jni.hsa_jni;

import moa.core.DoubleVector;
import weka.core.Instance;
import weka.core.SparseInstance;

public class SGD extends moa.classifiers.functions.SGD {
	
	
	class SparseInstanceAccess extends SparseInstance {
		public SparseInstanceAccess(Instance i) {
			super(i);
		}
		
		public int[] getIndexes()
		{
			return m_Indices;
		}
		
		public double[] getValues() 
		{
			return m_AttValues;
		}
	}
	
	protected  double product(Instance inst1, DoubleVector weights, int classIndex) {
		SparseInstanceAccess sparseAccess = new SparseInstanceAccess(inst1);
		int[] indices = sparseAccess.getIndexes();
		double[] values = sparseAccess.getValues();
		double result = 0;
		for (int i = 0 ;i < indices.length ; ++i)
			if (indices[i] != classIndex)
				result += values [ i ] * weights.getValue(indices[i]);
	
        return (result);
    }
	  /**
     * Trains the classifier with the given instance.
     *
     * @param instance 	the new training instance to include in the model
     */
    @Override
    public void trainOnInstanceImpl(Instance instance) {

        if (m_weights == null) {
            m_weights = new DoubleVector(); 
            m_bias = 0.0;
        }
        
        
        if (!instance.classIsMissing()) {

            double wx = product(instance, m_weights, instance.classIndex());

            double y;
            double z;
            if (instance.classAttribute().isNominal()) {
                y = (instance.classValue() == 0) ? -1 : 1;
                z = y * (wx + m_bias);
            } else {
                y = instance.classValue();
                z = y - (wx + m_bias);
                y = 1;
            }

            // Compute multiplier for weight decay
            double multiplier = 1.0;
            if (m_numInstances == 0) {
                multiplier = 1.0 - (m_learningRate * m_lambda) / m_t;
            } else {
                multiplier = 1.0 - (m_learningRate * m_lambda) / m_numInstances;
            }
            for (int i = 0; i < m_weights.numValues(); i++) {
                m_weights.setValue(i,m_weights.getValue (i) * multiplier);
            }

            // Only need to do the following if the loss is non-zero
            if (m_loss != HINGE || (z < 1)) {

                // Compute Factor for updates
                double factor = m_learningRate * y * dloss(z);

                // Update coefficients for attributes
                int n1 = instance.numValues();
                for (int p1 = 0; p1 < n1; p1++) {
                    int indS = instance.index(p1);
                    if (indS != instance.classIndex() && !instance.isMissingSparse(p1)) {
                        m_weights.addToValue(indS, factor * instance.valueSparse(p1));
                    }
                }

                // update the bias
                m_bias += factor;
            }
            m_t++;
        }
    }

    /**
     * Calculates the class membership probabilities for the given test
     * instance.
     *
     * @param instance 	the instance to be classified
     * @return 		predicted class probability distribution
     */
    @Override
    public double[] getVotesForInstance(Instance inst) {

        if (m_weights == null) {
            return new double[inst.numClasses()];
        }
        double[] result = (inst.classAttribute().isNominal())
                ? new double[2]
                : new double[1];


        double wx = product(inst, m_weights, inst.classIndex());// * m_wScale;
        double z = (wx + m_bias);

        if (inst.classAttribute().isNumeric()) {
            result[0] = z;
            return result;
        }

        if (z <= 0) {
            //  z = 0;
            if (m_loss == LOGLOSS) {
                result[0] = 1.0 / (1.0 + Math.exp(z));
                result[1] = 1.0 - result[0];
            } else {
                result[0] = 1;
            }
        } else {
            if (m_loss == LOGLOSS) {
                result[1] = 1.0 / (1.0 + Math.exp(-z));
                result[0] = 1.0 - result[1];
            } else {
                result[1] = 1;
            }
        }
        return result;
    }
}
