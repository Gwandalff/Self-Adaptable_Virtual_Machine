package miniJava.interpreter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import miniJava.interpreter.AbstractAdaptationModule.Strategy;
import miniJava.interpreter.decision.model.Goal;
import miniJava.interpreter.decision.model.Resource;
import miniJava.interpreter.miniJava.Value;

public class FeedbackLoop {
	private static List<AbstractAdaptationModule> modules = new ArrayList<>();
	private static List<Boolean> activation = new ArrayList<>();
	private static AdaptationContext context = new MiniJavaAdaptationContext();
	private static long time = 0;
	
	public static void registerModule(AbstractAdaptationModule module) {
		modules.add(module);
		activation.add(false);
		context.registerModule(module);
	}
	
	public static void reset() {
		context = new MiniJavaAdaptationContext();
		modules = new ArrayList<>();
		activation = new ArrayList<>();
		time = 0;
	}
	
	public static final boolean updateBefore(IAdaptationNode s, Object[] args) {
		if(loopTrigger(s)) {
			loop();
		}
		boolean proceed = true;
		for (int i = 0; i < modules.size(); i++) {
			if(activation.get(i)) {
				AbstractAdaptationModule m = modules.get(i);
				if(!m.isTargetedNode(s)) continue;
				if(m.callStrategy() == Strategy.BEFORE) {
					m.updateBefore(s, args);
				}
				if(m.callStrategy() == Strategy.AROUND) {
					proceed = m.updateBefore(s, args) && proceed;
				}
			}
		}
		return proceed;
	}
	
	public static final Value updateAfter(IAdaptationNode s, Object[] args, Value returned) {
		Value out = returned;
		for (int i = 0; i < modules.size(); i++) {
			if(activation.get(i)) {
				AbstractAdaptationModule m = modules.get(i);
				if(m.callStrategy() != Strategy.BEFORE) {
					if(m.updateAfter(s, args, returned)) {
						out = m.byPassResult();
					}
				}
			}
		}
		return out;
	}
	
	public final static boolean loopTrigger(IAdaptationNode s) {
		long delta = System.currentTimeMillis()-time;
		if(delta > 3000) {
			time = System.currentTimeMillis();
			return true;
		} else {
			return false;
		}
	}
	
	public final static void loop() {
		//System.out.println("MAPE-K LOOP");
		monitor();
		analyze();
		plan();
		execute();
	}
	
	private final static void monitor() {
		context.setUserConfig(context.loadUserConfig());
		List<Resource> resources = context.getResources();
		for (Resource resource : resources) {
			if (resource.ID.equals("CPU_PERCENT")) {
				try {
					String getCPUPercent = "/home/gjouneau/.getCPU.sh";
					Process getCPU = Runtime.getRuntime().exec(getCPUPercent);
					getCPU.waitFor();
					final BufferedReader is = new BufferedReader(new InputStreamReader(getCPU.getInputStream()));
		            String percentageCPU = is.readLine();
		            //System.out.println("CPU : " + percentageCPU);
					resource.setValue(Double.parseDouble(percentageCPU) / 100);
				} catch (IOException | InterruptedException e) {
					//System.out.println("PARSE ERROR");
					resource.setValue(0);
				}
			} else {
				resource.setValue(0);
			}
		}
	}
	
	private final static void analyze() {
		Goal tradeOff = context.getGoal();
		tradeOff.cleanModel();
		Map<String, Double> userConfig = context.getUserConfig();
		Set<String> IDS = userConfig.keySet();
		
		for (AbstractAdaptationModule module : modules) {
			for (String id : IDS) {
				module.getModuleTradeOff().updateLink(id, userConfig.get(id));
			}
		}
		
		context.getGoal().assessVariables();
	}
	
	private final static void plan   () {
		activation = new ArrayList<>();
		for (AbstractAdaptationModule module : modules) {
				activation.add(module.getModuleTradeOff().execute());
		}
	}
	
	private final static void execute() {
		for (int i = 0; i < modules.size(); i++) {
			modules.get(i).setEnabled(activation.get(i));
		}
	}

	public static AdaptationContext getAdaptationContext() {
		return context;
	}

}
