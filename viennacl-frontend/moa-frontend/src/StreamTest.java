

import java.io.FileInputStream;
import java.util.List;

import javax.xml.bind.JAXB;

import org.moa.gpu.Context;
import org.moa.gpu.config.Experiments;
import org.moa.gpu.config.SGDExperiment;

import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.MeasureStreamSpeed;
import moa.tasks.NullMonitor;

public class StreamTest {

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
				
				MeasureStreamSpeed speed = new MeasureStreamSpeed();
				speed.streamOption.setCurrentObject(generator);
				Object ret = speed.doTask(new NullMonitor(),null);
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

