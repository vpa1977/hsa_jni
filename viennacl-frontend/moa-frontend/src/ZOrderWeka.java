import weka.classifiers.lazy.IBk;

public class ZOrderWeka {
	public static void main(String[] args)
	{
		String testFile = "F:\\weka\\data\\ionosphere.arff";
		String outputFile = "output.txt";
	
                IBk linear = new IBk();
		IBk.runClassifier(linear, new String[]{
				"-t", testFile,
				"-d", outputFile,
				"-K", "3", "-W", "0", "-A", 
				"weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\" -P"
		});
		
		System.out.println("Search stats\n" + linear.getNearestNeighbourSearchAlgorithm().getPerformanceStats().getStats());
	
		IBk classifier = new IBk();
		IBk.runClassifier(classifier, new String[]{
				"-t", testFile,
				"-d", outputFile,
				"-K", "3", "-W", "0", "-A", 
				"weka.core.neighboursearch.ZOrderSearch -P -R -D 4 -C 16 -A \"weka.core.EuclideanDistance -R first-last\""
		});
		
		System.out.println("Search stats\n" + classifier.getNearestNeighbourSearchAlgorithm().getPerformanceStats().getStats());
	}
}
