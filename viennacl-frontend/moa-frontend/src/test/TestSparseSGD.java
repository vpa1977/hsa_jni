package test;

import java.io.FileInputStream;

import org.junit.Assert;
import org.junit.Test;
import org.moa.gpu.FastRandomTreeGenerator;
import org.moa.gpu.SparseSGD;

import weka.core.Instance;

public class TestSparseSGD {
	@Test
	public void testSideBySide() throws Throwable 
	{
		System.setProperty("java.library.path", "/home/bsp/git_repository/hsa_jni/viennacl-frontend/moa-frontend/");
		
		FileInputStream fis = new FileInputStream("libhsa-runtime64.so.1");
		fis.available();
		fis.close();
		
		System.loadLibrary("moa-frontend-lib");
		ArffStreamGenerator testStream = new ArffStreamGenerator();
		testStream.fileOption.setValue("test.arff");
		testStream.prepareForUse();
		
		
		FastRandomTreeGenerator stream = new FastRandomTreeGenerator();
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
