package weka.core.neighboursearch;

import java.io.Serializable;

public class FixedInt implements Comparable<FixedInt>, Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private long[] m_data; 
	public FixedInt(int num_bits, byte[] bits)
	{
		int long_size = bits.length / 8;
		long_size += long_size * 8 == bits.length ? 0 : 1;
		m_data = new long[long_size];
		for (int i = 0 ;i < long_size ; ++i)
			m_data[i] = makeLong(i,bits);
	}
	

	private long makeLong(int i, byte[] bits) {
		int offset = bits.length - i * 8 - 8;
		int byte_pos = 7;
		long result = 0;
		for (int pos = offset ; pos < offset + 8; ++pos, --byte_pos)
		{
			if (pos >= 0)
			{
				long temp = bits[pos];
				temp = temp << (byte_pos * 8);
				result |= (pos >= 0) ? temp : 0;
			}
		}
		return result;
	}


	@Override
	public int compareTo(FixedInt o) {
		if (o.m_data.length < m_data.length)
			return 1;
		if (o.m_data.length > m_data.length)
			return -1;
		for (int i = 0 ;i < o.m_data.length ; ++i)
		{
			if (m_data[i] < o.m_data[i] )
				return -1;
			if (m_data[i] > o.m_data[i] )
				return 1;	
		}	
		return 0;
	}

}
