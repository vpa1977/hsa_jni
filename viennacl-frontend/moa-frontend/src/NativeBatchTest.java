

import java.io.FileInputStream;
import java.util.List;

import javax.xml.bind.JAXB;

import org.moa.gpu.Context;
import org.moa.gpu.config.Experiments;
import org.moa.gpu.config.SGDExperiment;

import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.NullMonitor;
import test.TestNativeBatchPerformance;

public class NativeBatchTest {

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
				
								
				
				TestNativeBatchPerformance moaSGD = new test.TestNativeBatchPerformance();
				String generatorCLI = experiment.getGenerator();
				
				InstanceStream generator = (InstanceStream)ClassOption.cliStringToObject(generatorCLI,InstanceStream.class, null );
				
				
				int window = experiment.getWindow();
				int train_size = experiment.getTrainSize();
				int test_size = experiment.getTestSize();
				int train_batch = experiment.getTestBatch();
				int test_batch = experiment.getTrainBatch();
				
				
				EvaluatePeriodicHeldOutTest test = new EvaluatePeriodicHeldOutTest();
				test.learnerOption.setCurrentObject(moaSGD);
				test.streamOption.setCurrentObject(generator);
				test.testSizeOption.setValue(test_size);
				test.trainSizeOption.setValue(train_size);
				System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
				Object ret = test.doTask(new NullMonitor(),null);
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

