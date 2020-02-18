package miniJava.visitor.miniJava.impl;

import java.lang.Override;
import java.util.HashSet;
import java.util.Set;

import miniJava.visitor.miniJava.Call;
import miniJava.visitor.miniJava.MiniJavaPackage;
import visitor.IDynamicModule;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

public abstract class CallImpl extends MinimalEObjectImpl.Container implements Call {
	protected Set<IDynamicModule> modules;
	
	protected CallImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.CALL;
	}
	
	@Override
	public void attach(IDynamicModule dm) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void detach(IDynamicModule dm) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean notifyDynamicModulesBefore() {
		boolean out = true;
		for (IDynamicModule dm : modules) {
			out = out && dm.updateBefore(this);
		}
		return out;
	}

	@Override
	public void notifyDynamicModulesAfter() {
		for (IDynamicModule dm : modules) {
			dm.updateAfter(this);
		}
	}
}
