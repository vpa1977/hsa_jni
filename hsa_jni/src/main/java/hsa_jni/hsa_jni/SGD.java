package hsa_jni.hsa_jni;

import moa.core.DoubleVector;
import moa.options.IntOption;
import weka.core.Instance;

public class SGD extends moa.classifiers.functions.SGD implements
		BatchClassifier {
	
	
	private WekaHSAContext.SGD m_native_context = new WekaHSAContext().createNativeSGD();

	public IntOption trainBatchSizeOption = new IntOption("train_batch_size",
			'T', "Size of the training batch for the GPU offload", 100);
	public IntOption testBatchSizeOption = new IntOption("test_batch_size",
			't', "Size of the test batch for the GPU offload", 100);

	private InstanceBatch m_batch;
	
	protected double product(Instance inst, DoubleVector weights, int classIndex)
	{
		SparseInstanceAccess sparseAccess = new SparseInstanceAccess(inst);
		return product(sparseAccess.getValues(), sparseAccess.getIndexes(), weights, classIndex);
	}

	protected double product(double[] values, int[] indices, DoubleVector weights,
			int classIndex) {
		double result = 0;
		for (int i = 0; i < indices.length; ++i)
			if (indices[i] != classIndex)
				result += values[i] * weights.getValue(indices[i]);

		return (result);
	}

	/**
	 * Trains the classifier with the given instance.
	 *
	 * @param instance
	 *            the new training instance to include in the model
	 */
	@Override
	public void trainOnInstanceImpl(Instance instance) {
		if (m_weights == null) {
			m_batch = new InstanceBatch(instance.dataset(),testBatchSizeOption.getValue());
			m_weights = new DoubleVector();
			
			m_weights.addToValue(instance.dataset().numAttributes(), 0);// init double array size 
			m_bias = 0.0;
		}
		if (instance.classIsMissing())
			return;
		
		if (m_batch.add(instance))
			return;
		else {
			processBatch();
			m_batch.commit();
		}
		
	}

	private void processBatch(){
		// Compute multiplier for weight decay
		double multiplier = 1.0;
		if (m_numInstances == 0) {
			multiplier = 1.0 - (m_learningRate * m_lambda) / m_t;
		} else {
			multiplier = 1.0 - (m_learningRate * m_lambda) / m_numInstances;
		}
		
		m_batch.setMultiplier(multiplier);
		m_batch.m_learningRate = m_learningRate;
		m_batch.m_bias = m_bias;
		m_batch.m_t = m_t;
		
		m_native_context.UpdateWeights(m_batch, m_weights.getArrayRef());
		
		m_learningRate = m_batch.m_learningRate;
		m_bias = m_batch.m_bias;
		m_t = m_batch.m_t;
		
		/*
		 * // this for loop is for offloading
		 */
/*		for (int i = 0; i < m_batch.size(); ++i) {
			processInstance(m_batch.classValues()[i],
					        m_batch.values()[i], 
					        m_batch.indices()[i], 
					        m_batch.isNominal(), 
					        m_batch.classIndex());
		}
		*/
		
		for (int i = 0; i < m_weights.numValues(); i++) {
			m_weights.setValue(i, m_weights.getValue(i) * multiplier);
		}

	}

	private void processInstance(double class_value, double[] values, int[] indices, boolean is_nominal, int class_index) {
		double wx = product(values, indices, m_weights, class_index);

		double y;
		double z;
		if (is_nominal) {
			y = (class_value == 0) ? -1 : 1;
			z = y * (wx + m_bias);
		} else {
			y = class_value;
			z = y - (wx + m_bias);
			y = 1;
		}

		// Only need to do the following if the loss is non-zero
		if (m_loss != HINGE || (z < 1)) {

			// Compute Factor for updates
			double factor = m_learningRate * y * dloss(z);

			// Update coefficients for attributes
			for (int i = 0 ;i < indices.length ; ++i)
			{
				if (indices[i] == class_index)
					continue;
				m_weights.addToValue(indices[i],	factor * values[i]);
			}
			// update the bias
			m_bias += factor;
		}
		m_t++;
	}

	/**
	 * Calculates the class membership probabilities for the given test
	 * instance.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 */
	@Override
	public double[] getVotesForInstance(Instance inst) {

		if (m_weights == null) {
			return new double[inst.numClasses()];
		}
		double[] result = (inst.classAttribute().isNominal()) ? new double[2]
				: new double[1];

		double wx = product(inst, m_weights, inst.classIndex());// * m_wScale;
		double z = (wx + m_bias);

		if (inst.classAttribute().isNumeric()) {
			result[0] = z;
			return result;
		}

		if (z <= 0) {
			// z = 0;
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

	
	public void commit() {
		processBatch();
		m_batch.commit();

	}
}
