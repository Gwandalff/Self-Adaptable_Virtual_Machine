package miniJava.interpreter.dynamicModules;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.eclipse.emf.common.util.BasicEList;
import org.eclipse.emf.common.util.EList;
import org.eclipse.xtext.EcoreUtil2;

import miniJava.interpreter.IDynamicModule;
import miniJava.interpreter.IDynamicSubject;
import miniJava.interpreter.miniJava.Context;
import miniJava.interpreter.miniJava.Method;
import miniJava.interpreter.miniJava.MethodCall;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Parameter;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Symbol;
import miniJava.interpreter.miniJava.SymbolBinding;
import miniJava.interpreter.miniJava.Value;

public class DynamicMemoizationModule implements IDynamicModule {
	
	private Map<Method,Map<EList<Value>,Value>> memoizeTable = new HashMap();
	
	private Deque<Boolean> inMemory = new ArrayDeque<>();
	private Deque<EList<Value>> effectiveArguments = new ArrayDeque<>();
	private Value memoizedValue = null;

	@Override
	public boolean updateBefore(IDynamicSubject self, Object[] args) {
		MethodCall call = ((MethodCall) self);
		effectiveArguments.addLast(getEffectiveArguments((State) args[1], call.getMethod()));
		
		Map<EList<Value>,Value> table = memoizeTable.get(call.getMethod());		
		if(table == null) {
			memoizeTable.put(call.getMethod(), new HashMap<EList<Value>,Value>());
			inMemory.addLast(false);
		} else {
			Value val = getOutputFromParams(effectiveArguments.getLast(), table);
			inMemory.addLast(val != null);
		}
		
		return !inMemory.getLast();
	}
	
	@Override
	public boolean updateAfter(IDynamicSubject self, Object[] args, Value returned) {
		MethodCall call = ((MethodCall) self);
		Map<EList<Value>,Value> table = memoizeTable.get(call.getMethod());
		if(!inMemory.getLast()) {
			Value trueReturned = ((State) args[1]).findCurrentFrame().getReturnValue();
			table.put(effectiveArguments.removeLast(), EcoreUtil2.copy(trueReturned));
			memoizeTable.put(call.getMethod(), table);
		} else {
			memoizedValue = getOutputFromParams(effectiveArguments.removeLast(), table);
			((State) args[1]).findCurrentFrame().setReturnValue(memoizedValue);
		}
		inMemory.removeLast();
		return false;
	}
	
	private EList<Value> getEffectiveArguments(State state, Method method){
		Context ctx = state.findCurrentContext();
		EList<SymbolBinding> bindings = ctx.getBindings();
		EList<Value> effectiveParameters = new BasicEList();
		for (Parameter param : method.getParams()) {
			for (int i = 0; i < bindings.size(); i++) {
				Symbol s = bindings.get(i).getSymbol();
				if(s instanceof Parameter && ((Parameter) s).compare(param)) {
					effectiveParameters.add(bindings.get(i).getValue());
				}
			}
		}
		return effectiveParameters;
	}
	
	private Value getOutputFromParams(EList<Value> args, Map<EList<Value>,Value> table){
		Set<EList<Value>> keys = table.keySet();
		EList<Value> theKey = null;
		for (EList<Value> key : keys) {
			boolean valid = true;
			for (int i = 0; i < key.size(); i++) {
				valid = valid & key.get(i).equals(args.get(i));
				if(!valid) break;
			}
			if(valid) {
				theKey = key;
				break;
			}
		}
		return theKey==null?null:EcoreUtil2.copy(table.get(theKey));
	}

	@Override
	public Value byPassResult() {
		return memoizedValue;
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
