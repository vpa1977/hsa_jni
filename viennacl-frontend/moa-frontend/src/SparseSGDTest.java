

import org.moa.gpu.ConsoleMonitor;
import org.moa.gpu.Context;
import org.moa.gpu.DenseSGD;
import org.moa.gpu.SparseSGD;
import org.moa.gpu.SimpleDirectMemoryBatchInstances;
import org.moa.gpu.SimpleWindow;
import org.moa.gpu.config.*;
import org.moa.gpu.util.EvaluateTrainSpeed;

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

public class SparseSGDTest {

/**
	 * @param args
	 */
	public static void main(String[] args) throws Throwable {
		System.out.println( System.getProperty("java.class.path"));
		Context.load();
		
		RandomTreeGenerator d;
		
		
		Experiments experiments = JAXB.unmarshal(new FileInputStream(args[0]),Experiments.class );
		
		List<SGDExperiment> list = experiments.getSgd();
		
		for (SGDExperiment experiment : list )
		{
			try {
				
								
				
				test.SGD moaSGD = new test.SGD();
				String generatorCLI = experiment.getGenerator();
				
				InstanceStream generator = (InstanceStream)ClassOption.cliStringToObject(generatorCLI,InstanceStream.class, null );
				
				
				
				int train_size = experiment.getTrainSize();
				int test_size = experiment.getTestSize();
				int train_batch = experiment.getTrainBatch();
				int test_batch = experiment.getTestBatch();
				
				
				SparseSGD hsaSGD = new SparseSGD();
				hsaSGD.learningRateOption.setValue(0.000001);
				Object ret;
				
				//System.out.println("Test : window="+ window);
				//System.out.println("     : test_batch="+ test_batch);
				//System.out.println("     : train_batch="+ train_batch);
				
				System.out.println("	 : test_size="+ test_size + " train_size =" + train_size + " train batch="+train_batch);
				System.out.println("Stream: "+ ((AbstractOptionHandler)generator).getCLICreationString(InstanceStream.class));// + " "+ generator.getOptions().getAsCLIString());
				
				hsaSGD.learningBatchSize.setValue(train_batch);
				EvaluateTrainSpeed test ;
				
				test = new EvaluateTrainSpeed();
				test.trainTimeOption.setValue(60);
				test.sampleFrequencyOption.setValue(train_batch*5+1);

				test.streamOption.setCurrentObject(generator);
				test.learnerOption.setCurrentObject(hsaSGD);
				test.testSizeOption.setValue(test_size);
				test.trainSizeOption.setValue(train_size);
				System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
				 ret = test.doTask(new NullMonitor(),null);
				System.out.println(ret);
				System.out.println("---------------------------------------------------------------------------");
				
				test = new EvaluateTrainSpeed();
				test.trainTimeOption.setValue(60);
				test.sampleFrequencyOption.setValue(1000);
				test.learnerOption.setCurrentObject(moaSGD);
				test.streamOption.setCurrentObject(generator);
				test.testSizeOption.setValue(test_size);
				test.trainSizeOption.setValue(train_size);
				System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
				ret = test.doTask(new ConsoleMonitor(),null);
				System.out.println(ret);
				System.out.println("---------------------------------------------------------------------------");
				System.exit(0);
			}
			catch (Throwable t )
			{
				t.printStackTrace();
			}

		}
		
		
		 
		
		
	}

}

