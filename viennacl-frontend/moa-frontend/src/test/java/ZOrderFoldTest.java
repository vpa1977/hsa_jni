package test.java;

import org.junit.Test;

import junit.framework.Assert;
import weka.core.neighboursearch.ZOrderFold;

public class ZOrderFoldTest {

	@Test
	public void testFold() {
		ZOrderFold fold = new ZOrderFold(0,11, 3,1);
		Assert.assertEquals(fold.mapping(0,0).size(), 3);
		Assert.assertEquals(fold.mapping(0,1).size(), 3);
		Assert.assertEquals(fold.mapping(0,2).size(), 4);
	}

}
