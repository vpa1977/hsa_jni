package org.moa.gpu;

import weka.core.DenseInstance;
import weka.core.Instance;
import moa.core.InstancesHeader;
import moa.streams.generators.RandomTreeGenerator;

public class FastRandomTreeGenerator extends RandomTreeGenerator {

	@Override
	public Instance nextInstance() {
        double[] attVals = new double[this.numNominalsOption.getValue()
                                      + this.numNumericsOption.getValue() +1];
	      InstancesHeader header = getHeader();
	      
	      for (int i = 0; i < attVals.length; i++) {
	          attVals[i] = i < this.numNominalsOption.getValue() ? this.instanceRandom.nextInt(this.numValsPerNominalOption.getValue())
	                  : this.instanceRandom.nextDouble();
	      }
	      Instance inst = new DenseInstance(1, attVals);
	      inst.setDataset(header);
	      inst.setClassValue(classifyInstance(this.treeRoot, attVals));
	      return inst;
	}

}
