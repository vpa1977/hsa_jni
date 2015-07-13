package org.moa.gpu;

import moa.tasks.NullMonitor;

public class ConsoleMonitor extends NullMonitor{

	@Override
	public void setCurrentActivity(String activityDescription,
			double fracComplete) {
		// TODO Auto-generated method stub
		super.setCurrentActivity(activityDescription, fracComplete);
		System.out.println(activityDescription+ " = " + fracComplete);
	}

	@Override
	public void setCurrentActivityFractionComplete(double fracComplete) {
		// TODO Auto-generated method stub
		super.setCurrentActivityFractionComplete(fracComplete);
		if (Math.abs( fracComplete - prev_frac) > 0.1)
			System.out.println(fracComplete);
		prev_frac = fracComplete;
	}
	
	double prev_frac = 0;

}
