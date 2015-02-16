package hsa_jni.hsa_jni;


import java.util.ArrayList;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

/***
 * Used to test if output from 2 different implementations match
 * @author ���
 *
 */
public class DualClassifier extends AbstractClassifier{
	
	private AbstractClassifier m_classifier1;
	private AbstractClassifier m_classifier2;
	private ArrayList<Instance> m_instances = new ArrayList<Instance>();
	
	public DualClassifier(AbstractClassifier c1, AbstractClassifier c2)
	{
		m_classifier1 = c1;
		m_classifier2 = c2;
				
		
	}


	public boolean isRandomizable() {
		return false;
	}

	
	public double[] getVotesForInstance(Instance inst) {
		double[] votes1 = m_classifier1.getVotesForInstance(inst);
		double[] votes2 = m_classifier2.getVotesForInstance(inst);
		
		for (int i = 0 ;i < votes1.length ; i ++)
		{
			if (votes1[i]!=votes2[i])
			{
				m_classifier1.getVotesForInstance(inst);
				
				m_classifier2.getVotesForInstance(inst);
				//throw new RuntimeException("ops");
				
				
			}
		}
		return votes1;
	}

	@Override
	public void resetLearningImpl() {
		m_classifier1.resetLearningImpl();
		m_classifier2.resetLearningImpl();
		m_instances.clear();
		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		m_classifier1.trainOnInstanceImpl(inst);
		m_classifier2.trainOnInstanceImpl(inst);
		m_instances.add(inst);
		
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