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
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
/**
 * Generate zOrder representation of the dataset 
 * using projected dataset.
 * @author bsp
 */
public class ZOrder implements Serializable {
	
	
	
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	
    /**
     * convert double to long bitwise representation
     * @param value
     * @return 
     */
    protected long doubleAsLong(double input, double min, double width)
    {
    	double test = (input - min);
    	if (width !=0)
    		test = test/width;
    	//System.out.println("Test values "+ input + " normalized "+ test + "adjusted "+ (long)(test * Long.MAX_VALUE));
    	return Double.doubleToLongBits(test);
        //long value = Double.doubleToRawLongBits(input);
        //long res = value ^ (-(value >> 63) | 0x8000000000000000L);
        //return res;
    	//return Float.floatToIntBits((float)input);
    }
    

	/** 
     * Interleave bits of the weka instance
     * @param instance
	 * @param z_order_norm 
     * @param curve - projection + random shift if enabled -  
     * @return 
     */
    public FixedInt interleave(Instance instance, NormalizableDistance z_order_norm) throws Exception
    {
    	return interleaveAsLong(instance, z_order_norm);
    }
    
    private FixedInt interleaveAsInt(Instance instance, NormalizableDistance z_order_norm) throws Exception {
		int num_attributes = instance.numAttributes() -1;
        byte[] data = new byte[num_attributes * 4]; // 32 bit per each attribute
                                                                    // x number of attributes
        for (int bit = 0 ; bit < data.length * 4 ; ++bit)
        {
           // retrieve bit from the attribute
           int attribute =  bit % num_attributes;
           
           int offset = bit / num_attributes;
           if (attribute >= instance.classIndex())
               attribute += 1;
           double attr_value = 0;
           
           
           attr_value = instance.value(attribute);
           double[][] ranges = z_order_norm.getRanges();
           int int_value = doubleAsInt(attr_value, ranges[attribute][NormalizableDistance.R_MIN], ranges[attribute][NormalizableDistance.R_WIDTH]);
           long bit_value =  (int_value & (1 << offset)) >> offset;
           // set bit in the array
           int byte_position = bit / 8;
           int bit_position = bit % 8;
           data[data.length -1 - byte_position] |=  (bit_value << bit_position);
           //data[byte_position] |=  (bit_value << bit_position);
        }
        return new FixedInt(data.length*8, data);
    	
    }


	private int doubleAsInt(double input, double min, double width) {
    	double test = (input - min);
    	if (width !=0)
    		test = test/width;
    	//System.out.println("Test values "+ input + " normalized "+ test + "adjusted "+ (long)(test * Long.MAX_VALUE));
    	return Float.floatToIntBits((float)test);	
    }


	private FixedInt interleaveAsLong(Instance instance, NormalizableDistance z_order_norm) throws Exception {
		int num_attributes = instance.numAttributes() -1;
        byte[] data = new byte[num_attributes * 8]; // 64 bit per each attribute
                                                                    // x number of attributes
        for (int bit = 0 ; bit < data.length * 8 ; ++bit)
        {
           // retrieve bit from the attribute
           int attribute =  bit % num_attributes;
           
           int offset = bit / num_attributes;
           if (attribute >= instance.classIndex())
               attribute += 1;
           double attr_value = 0;
           
           
           attr_value = instance.value(attribute);
           double[][] ranges = z_order_norm.getRanges();
           long long_value = doubleAsLong(attr_value, ranges[attribute][NormalizableDistance.R_MIN], ranges[attribute][NormalizableDistance.R_WIDTH]);
           long bit_value =  (long_value & (1 << offset)) >> offset;
           // set bit in the array
           int byte_position = bit / 8;
           int bit_position = bit % 8;
           data[data.length -1 - byte_position] |=  (bit_value << bit_position);
           //data[byte_position] |=  (bit_value << bit_position);
        }
        return new FixedInt(data.length* 8,data);
	}
    
    /*
     * create z-order based on projection dataset 
     */
    public ArrayList< ZOrderInstance > createZOrder(Instances dataset, Instances source)
    {
    	EuclideanDistance z_order_norm = new EuclideanDistance();
    	z_order_norm.setInstances(dataset);
        ArrayList<ZOrderInstance> z_order = new ArrayList<ZOrderInstance>();
        int size = dataset.numInstances();
        for (int i = 0; i < size; ++i)
        {
            Instance inst = dataset.instance(i);
            try {
            	FixedInt value = interleave(inst, z_order_norm);
            	z_order.add(new ZOrderInstance(value, i));
            }
            catch (Exception e) { e.printStackTrace(); }
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
