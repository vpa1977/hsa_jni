package test.java;

import static org.junit.Assert.*;

import org.junit.Test;

import junit.framework.Assert;
import weka.core.neighboursearch.ZOrderFold;

public class ZOrderFoldTest {

	@Test
	public void testFold() {
		ZOrderFold fold = new ZOrderFold(0,11, 3);
		Assert.assertEquals(fold.mapping(0).size(), 3);
		Assert.assertEquals(fold.mapping(1).size(), 3);
		Assert.assertEquals(fold.mapping(2).size(), 4);
	}

}
