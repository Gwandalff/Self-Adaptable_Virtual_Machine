package miniJava.interpreter;

import java.util.HashMap;
import java.util.Map;

import miniJava.interpreter.decision.model.Resource;

public class MiniJavaAdaptationContext extends AdaptationContext {
	
	public MiniJavaAdaptationContext() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public Map<String, Double> loadUserConfig() {
		Map<String, Double> out = new HashMap<String, Double>();
		out.put("Performance", Double.parseDouble(System.getenv("ADAPTABLE_PERFORMANCE")) / 100.0);
		out.put("Accuracy", Double.parseDouble(System.getenv("ADAPTABLE_ACCURACY")) / 100.0);
		return out;
	}

	@Override
	public String[] softgoalIDs() {
		return new String[] {"Performance", "Accuracy"};
	}

	@Override
	protected Resource[] createResources() {
		return new Resource[] {new Resource("CPU_PERCENT")};
	}

}
