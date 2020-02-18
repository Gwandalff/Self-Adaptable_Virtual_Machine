package visitor.dynamicmodules;

import java.util.HashMap;
import java.util.Map;

import org.eclipse.emf.common.util.EList;

import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.MethodCall;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Value;
import visitor.AcceptInterface;
import visitor.IDynamicModule;

public class DynamicMemoizationModule implements IDynamicModule {
	
	Map<Method,Map<EList<Expression>,Value>> memoizeTable = new HashMap();

	@Override
	public boolean updateBefore(AcceptInterface self) {
		System.out.println("module memoize");
		MethodCall call = ((MethodCall) self);
		Map<EList<Expression>,Value> table = memoizeTable.get(call.getMethod());
		if(table == null) {
			memoizeTable.put(call.getMethod(), new HashMap<EList<Expression>,Value>());
		} else {
			Value val = table.get(call.getArgs());
			if(val == null) {
				// wait and register value
			} else {
				// return memoized value
			}
		}
		
		return true;
	}
	
	@Override
	public void updateAfter(AcceptInterface self) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int[] targetedNodes() {
		int[] out = {MiniJavaPackage.METHOD_CALL};
		return out;
	}
	
	@Override
	public Strategy callStrategy() {
		return Strategy.AROUND;
	}

}
