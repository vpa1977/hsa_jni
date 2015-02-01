package hsa_jni.hsa_jni;



import hsa_jni.hsa_jni.config.Experiments;
import hsa_jni.hsa_jni.config.FirstKnnExperiment;
import hsa_jni.hsa_jni.config.SGDExperiment;

import java.io.FileInputStream;
import java.util.List;

import javax.xml.bind.JAXB;

import weka.classifiers.lazy.IBk;
import weka.core.neighboursearch.LinearNNSearch;
import moa.classifiers.meta.WEKAClassifier;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.NullMonitor;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.JavaCL;

public class DualSGDTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Throwable {
		
		
		RandomTreeGenerator d;
		
		WekaHSAContext context = new WekaHSAContext();
		
		Experiments experiments = JAXB.unmarshal(new FileInputStream(args[0]),Experiments.class );
		
		List<SGDExperiment> list = experiments.getSgd();
		
		for (SGDExperiment experiment : list )
		{
			try {
				
				moa.classifiers.functions.SGD moaSGD = new moa.classifiers.functions.SGD();
				String generatorCLI = experiment.getGenerator();
				
				InstanceStream generator = (InstanceStream)ClassOption.cliStringToObject(generatorCLI,InstanceStream.class, null );
				
				
				int window = experiment.getWindow();
				int train_size = experiment.getTrainSize();
				int test_size = experiment.getTestSize();
				int train_batch = experiment.getTestBatch();
				int test_batch = experiment.getTrainBatch();
				SGD hsaSGD = new SGD();
				hsaSGD.trainBatchSizeOption.setValue(train_batch);
				hsaSGD.testBatchSizeOption.setValue(test_batch);
				
				System.out.println("Test : window="+ window);
				System.out.println("     : test_batch="+ test_batch);
				System.out.println("     : train_batch="+ train_batch);
				
				System.out.println("	 : test_size="+ test_size + " train_size =" + train_size);
				System.out.println("Stream: "+ ((AbstractOptionHandler)generator).getCLICreationString(InstanceStream.class));// + " "+ generator.getOptions().getAsCLIString());
				
				
				
				
				EvaluatePeriodicHeldOutTestBatch test = new EvaluatePeriodicHeldOutTestBatch();
				test.streamOption.setCurrentObject(generator);
				test.learnerOption.setCurrentObject(hsaSGD);
				test.testSizeOption.setValue(test_size);
				test.trainSizeOption.setValue(train_size);
				System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
				Object ret = test.doTask(new NullMonitor(),null);
				System.out.println(ret);
				System.out.println("---------------------------------------------------------------------------");
				
				test = new EvaluatePeriodicHeldOutTestBatch();
				test.learnerOption.setCurrentObject(moaSGD);
				test.streamOption.setCurrentObject(generator);
				test.testSizeOption.setValue(test_size);
				test.trainSizeOption.setValue(train_size);
				System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
				ret = test.doTask(new NullMonitor(),null);
				System.out.println(ret);
				System.out.println("---------------------------------------------------------------------------");

			}
			catch (Throwable t )
			{
				t.printStackTrace();
			}

		}
		
		
		 
		
		
	}

}

