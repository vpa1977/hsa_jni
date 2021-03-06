

import java.io.FileInputStream;
import java.util.List;

import javax.xml.bind.JAXB;

import org.moa.gpu.Context;
import org.moa.gpu.SparseSGD;
import org.moa.gpu.config.Experiments;
import org.moa.gpu.config.SGDExperiment;
import org.moa.gpu.util.EvaluateTrainSpeed;

import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.NullMonitor;

public class DualSparseSGDTest {

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
				
				
				
				SparseSGD hsaSGD = new SparseSGD();
				hsaSGD.learningRateOption.setValue(0.000001);
				hsaSGD.learningBatchSize.setValue(train_batch);
				Object ret;
				
				//System.out.println("Test : window="+ window);
				//System.out.println("     : test_batch="+ test_batch);
				//System.out.println("     : train_batch="+ train_batch);
				
				System.out.println("	 : test_size="+ test_size + " train_size =" + train_size + " train batch="+train_batch);
				System.out.println("Stream: "+ ((AbstractOptionHandler)generator).getCLICreationString(InstanceStream.class));// + " "+ generator.getOptions().getAsCLIString());
				
				
				EvaluateTrainSpeed test ;
				if (true)
				{
				
				test = new EvaluateTrainSpeed();
				test.trainTimeOption.setValue(60);
				test.sampleFrequencyOption.setValue(train_batch*2+1);
				test.streamOption.setCurrentObject(generator);
				test.learnerOption.setCurrentObject(hsaSGD);
				test.testSizeOption.setValue(test_size);
				test.trainSizeOption.setValue(train_size);
				System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
				 ret = test.doTask(new NullMonitor(),null);
				System.out.println(ret);
				System.out.println("---------------------------------------------------------------------------");
				}
				test = new EvaluateTrainSpeed();
				test.trainTimeOption.setValue(60);
				test.sampleFrequencyOption.setValue(5000);
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

