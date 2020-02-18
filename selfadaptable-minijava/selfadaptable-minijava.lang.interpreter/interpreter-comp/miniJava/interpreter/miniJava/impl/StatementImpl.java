package miniJava.interpreter.miniJava.impl;

import java.util.HashSet;
import java.util.Set;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Statement;
import miniJava.interpreter.miniJava.Value;
import miniJava.interpreter.IDynamicModule;
import miniJava.interpreter.IDynamicModule.Strategy;

public class StatementImpl extends MinimalEObjectImpl.Container implements Statement {
	protected Set<IDynamicModule> modules;
	
	protected StatementImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.STATEMENT;
	}

	public void evaluateStatement(State state) {
	}
	
	@Override
	public void attach(IDynamicModule dm) {
		modules.add(dm);
	}

	@Override
	public void detach(IDynamicModule dm) {
		modules.remove(dm);
	}

	@Override
	public boolean notifyDynamicModulesBefore(Object[] args) {
		boolean out = true;
		for (IDynamicModule dm : modules) {
			if(dm.callStrategy() == Strategy.AROUND) {
				out = out && dm.updateBefore(this, args);
			} else {
				dm.updateBefore(this, args);
			}
				
		}
		return out;
	}

	@Override
	public Value notifyDynamicModulesAfter(Object[] args, Value result) {
		Value out = result;
		for (IDynamicModule dm : modules) {
			if(dm.updateAfter(this, args, result)) {
				out = dm.byPassResult();
			}				
		}
		return out;
	}
}
