package org.moa.gpu;

public class Main {

	public static void main(String[] args) throws Throwable {
		System.out.println(System.getProperty("java.class.path"));
		Context.load();
		SGD test = new SGD(new SimpleWindow(122),0);
		test.getVotesForInstance(null);
		
	}

}
