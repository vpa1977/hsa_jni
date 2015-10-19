/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.core.neighboursearch;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
/**
 * Generate zOrder representation of the 
 * instance
 * @author bsp
 */
public class ZOrder implements Serializable {
	
	private Instance[] m_randomizers;
	private int m_num_dimensions;
	private ZOrderFold m_dimension_reduction;
	private boolean m_random_shift;
	private NormalizableDistance m_DistanceFunction;
    
	public ZOrder(NormalizableDistance distance_function,boolean random_shift, int num_attributes, int num_random_vectors)
	{
		init(num_attributes, num_random_vectors);
		m_num_dimensions = num_attributes-1;
		m_random_shift = random_shift;
		m_DistanceFunction = distance_function;
	}
	
	public ZOrder(NormalizableDistance distance_function,boolean random_shift, int num_attributes, int num_random_vectors, int class_index, int num_dimensions)
	{
		init(num_attributes, num_random_vectors);
		m_num_dimensions = num_dimensions;
		m_dimension_reduction = new ZOrderFold(class_index, num_attributes, num_dimensions);
		m_random_shift = random_shift;
		m_DistanceFunction = distance_function;
	}
	
	
	private void init(int num_attributes, int num_random_vectors)
	{
		Random rnd = new Random();
		m_randomizers = new Instance[num_random_vectors];
		for (int i = 0; i < num_random_vectors; ++i)
		{
			m_randomizers[i] = new DenseInstance(num_attributes);
			for (int j = 0 ; j < num_attributes ; ++j)
			{
				m_randomizers[i].setValue(j, rnd.nextDouble());
			}
		}
	}
	
    /**
     * convert double to long bitwise representation
     * @param value
     * @return 
     */
    protected long doubleAsLong(double input)
    {
        long value = Double.doubleToRawLongBits(input);
        long res = value ^ (-(value >> 63) | 0x8000000000000000L);
        return res;
    	//return Float.floatToIntBits((float)input);
    }
    
    public double fold(int attribute, Instance instance, int rot)
    {
        if (!m_DistanceFunction.rangesSet())
        {
        	m_DistanceFunction.initializeRanges();
        }

    	if (m_dimension_reduction == null)
    	{
            if (attribute >= instance.classIndex())
                attribute += 1;
            double val= 0;
            try {
				double[][] ranges = m_DistanceFunction.getRanges();
				val = (instance.value(attribute) - ranges[attribute][NormalizableDistance.R_MIN]) / (ranges[attribute][NormalizableDistance.R_WIDTH]);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
            return val + rot > m_randomizers.length && m_random_shift ? m_randomizers[rot].value(attribute) : 0;
    	}
    	else
    	{
    		double val = 0;
			try {
				double[][] ranges = m_DistanceFunction.getRanges();
				val = m_dimension_reduction.fold(ranges,attribute,instance);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    		return val+ rot > m_randomizers.length && m_random_shift ? m_randomizers[rot].value(attribute) : 0;
    	}
    }
    
    /** 
     * Interleave bits of the weka instance
     * @param instance
     * @param rot - rotation 
     * @return 
     */
    public BigInteger interleave(Instance instance, int rot)
    {
    	int num_attributes = m_num_dimensions;
        byte[] data = new byte[num_attributes * 8]; // 64 bit per each attribute
                                                                    // x number of attributes
        for (int bit = 0 ; bit < data.length * 8 ; ++bit)
        {
           // retrieve bit from the attribute
           int attribute =  bit % num_attributes;
           
           int offset = bit / num_attributes;
           double attr_value =fold(attribute, instance, rot);
           long long_value = doubleAsLong(attr_value);
           long bit_value =  (long_value & (1 << offset)) >> offset;
           // set bit in the array
           int byte_position = bit / 8;
           int bit_position = bit % 8;
           data[data.length -1 - byte_position] |=  (bit_value << bit_position);
           //data[byte_position] |=  (bit_value << bit_position);
        }
        return new BigInteger(data);
    }
    
    public ArrayList< ZOrderInstance > createZOrder(Instances dataset, int rot)
    {
        ArrayList<ZOrderInstance> z_order = new ArrayList<ZOrderInstance>();
        int size = dataset.numInstances();
        for (int i = 0; i < size; ++i)
        {
            Instance inst = dataset.instance(i);
            BigInteger value = interleave(inst, rot);
            z_order.add(new ZOrderInstance(value, inst));
        }
        Collections.sort(z_order, new Comparator<ZOrderInstance>() {
            @Override
            public int compare(ZOrderInstance o1, ZOrderInstance o2) {
                return o1.compareTo(o2);
            }
        });
        return z_order;
    }
    
    
    
}
