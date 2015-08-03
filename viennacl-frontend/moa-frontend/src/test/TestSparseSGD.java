package test;

import org.junit.Assert;
import org.junit.Test;
import org.moa.gpu.DenseSGD;
import org.moa.gpu.NativeDenseInstanceStreamGenerator;
import org.moa.gpu.SparseSGD;

import moa.streams.generators.RandomRBFGenerator;
import weka.core.Instance;

public class TestSparseSGD {
	@Test
	public void testSideBySide() 
	{
		System.loadLibrary("moa-frontend-lib");
		ArffStreamGenerator testStream = new ArffStreamGenerator();
		testStream.fileOption.setValue("test.arff");
		testStream.prepareForUse();
		
		
		NativeDenseInstanceStreamGenerator stream = new NativeDenseInstanceStreamGenerator();
		stream.baseStreamOption.setCurrentObject(testStream);
		stream.prepareForUse();
		
		test.SGD moaSGD = new test.SGD();
		moaSGD.prepareForUse();
		
		SparseSGD hsaSGD = new SparseSGD();
		hsaSGD.learningRateOption.setValue(moaSGD.learningRateOption.getValue());
		hsaSGD.learningBatchSize.setValue(1);
		hsaSGD.prepareForUse();

		for (int l = 0;l< 1000 ; l++)
		{
			Instance sample = stream.nextInstance();
			Instance test = stream.nextInstance();
			
			
			moaSGD.trainOnInstance(sample);
			
			hsaSGD.trainOnInstance(sample);
			
			
			moaSGD.printWeights();
			
			double[] hsaResult = hsaSGD.getVotesForInstance(test);
			double[] moaResult = moaSGD.getVotesForInstance(test);
			Assert.assertEquals(hsaResult.length, moaResult.length);
			for (int i = 0 ;i < moaResult.length ; ++i)
				Assert.assertEquals(hsaResult[i], moaResult[i], 0.01);
		}
		
	}
	
	

}
