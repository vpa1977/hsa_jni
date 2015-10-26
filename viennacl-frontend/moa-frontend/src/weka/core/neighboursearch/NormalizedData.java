package weka.core.neighboursearch;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;

public class NormalizedData implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private NormalizableDistance m_distance;
	private Instances m_data;
	private Instances m_src;

	public NormalizedData(NormalizableDistance dist) {
		m_distance = dist;
	}

	public void update(Instances src) throws Exception {
		double[][] ranges = m_distance.getRanges();
		Instances copy = new Instances(src);
		for (Instance inst : copy) {
			for (int i = 0; i < inst.numAttributes(); ++i) {
				if (!inst.isMissing(i))
					inst.setValue(i, (inst.value(i) - ranges[i][NormalizableDistance.R_MIN])
							/ ranges[i][NormalizableDistance.R_WIDTH]);
			}
		}
		m_data = copy;
		m_src = src;
	}

	public void update(Instance src) throws Exception {
		m_data.add(normalize(src));
	}

	public Instances normalizedData() {
		//return m_data;
		return m_src;
	}

	public Instance normalize(Instance src) throws Exception {
		if (true) return src;
		
		double[][] ranges = m_distance.getRanges();
		if (m_data == null)
			throw new NullPointerException("Missing dataset");
		Instance inst = (Instance) src.copy();
		for (int i = 0; i < inst.numAttributes(); ++i) {
			if (!inst.isMissing(i) && i!=inst.classIndex())
				inst.setValue(i, (inst.value(i) - ranges[i][NormalizableDistance.R_MIN])
						/ ranges[i][NormalizableDistance.R_WIDTH]);
		}
		return inst;
	}
}
