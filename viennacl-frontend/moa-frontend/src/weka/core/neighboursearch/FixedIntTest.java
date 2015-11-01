package weka.core.neighboursearch;

import static org.junit.Assert.*;

import org.junit.Test;

public class FixedIntTest {

	@Test
	public void testCreateFixedInt() {
		byte[] sample = { 0x1,0x2, 0x3, 0x4, 0x5, 0x6, 0x7,0x8, 0x9, 0xA, 0xB};
		FixedInt intTest = new FixedInt(sample.length *8, sample);
		System.out.println("stop");
	}

}
