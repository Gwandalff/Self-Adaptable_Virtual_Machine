package miniJava.interpreter.dynamicModules;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.eclipse.emf.ecore.EObject;

import miniJava.interpreter.IDynamicModule;
import miniJava.interpreter.IDynamicSubject;
import miniJava.interpreter.miniJava.Block;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.Method;
import miniJava.interpreter.miniJava.MethodCall;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.PrintStatement;
import miniJava.interpreter.miniJava.StringConstant;
import miniJava.interpreter.miniJava.Value;

public class DynamicPEModule implements IDynamicModule {
	
	private Map<String,Integer> callTable = new HashMap();
	private Map<String,Integer> computeTable = new HashMap();

	@Override
	public boolean updateBefore(IDynamicSubject self, Object[] args) {
		if(self instanceof PrintStatement) return false;
		if(self instanceof Block) {
			EObject container = ((Block) self).eContainer();
			if(container instanceof Method) {
				Method method = (Method) container;
				Integer count = computeTable.get(method.getName());
				if(count == null) {
					count = 0;
				}
				count += 1;
				computeTable.put(method.getName(), count);
			}
		} else {
			MethodCall call = ((MethodCall) self);
			Integer count = callTable.get(call.getMethod().getName());
			if(count == null) {
				count = 0;
			}
			count += 1;
			callTable.put(call.getMethod().getName(), count);
		}
		
		return true;
	}

	@Override
	public boolean updateAfter(IDynamicSubject self, Object[] args, Value returned) {		
		try {
			Expression exp = ((PrintStatement) self).getExpression();
			String print = ((StringConstant) exp).getValue();
			if(print.equals(":statistique")) {
				Set<String> keys = callTable.keySet();
				System.out.println("Overview of method calls :\n");
				for (String m : keys) {
					System.out.println("Method : "+m);
					System.out.println("\tCalled " + callTable.get(m) + " times");
					Integer computed = computeTable.get(m);
					if(computed != null) {
						System.out.println("\tComputed " + computed + " times");
					} else {
						System.out.println("\tNot Computed");
					}
				}
			}
		} catch (Exception e) {
			return false;
		}
		
		return false;
	}

	@Override
	public Value byPassResult() {
		return null;
	}

	@Override
	public int[] targetedNodes() {
		int[] out = {MiniJavaPackage.METHOD_CALL, MiniJavaPackage.BLOCK, MiniJavaPackage.PRINT_STATEMENT};
		return out;
	}
	
	@Override
	public Strategy callStrategy() {
		return Strategy.BEFORE;
	}

}
