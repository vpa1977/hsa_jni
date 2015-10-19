import weka.classifiers.lazy.IBk;

public class ZOrderWeka {
	public static void main(String[] args)
	{
		String testFile = "F:\\weka\\data\\ionosphere.arff";
		String outputFile = "output.txt";
		/*IBk.runClassifier(new IBk(), new String[]{
				"-t", testFile,
				"-d", outputFile,
				"-K", "3", "-W", "0", "-A", 
				"weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""
		});*/
		
		IBk.runClassifier(new IBk(), new String[]{
				"-t", testFile,
				"-d", outputFile,
				"-K", "3", "-W", "0", "-A", 
				"weka.core.neighboursearch.ZOrderSearch -R -D 5 -C 5 -A \"weka.core.EuclideanDistance -R first-last\""
		});

	}
}
