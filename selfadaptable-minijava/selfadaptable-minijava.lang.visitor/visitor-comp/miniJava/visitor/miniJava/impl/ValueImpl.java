package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.HashSet;
import java.util.Set;

import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class ValueImpl extends MinimalEObjectImpl.Container implements Value {
	protected Set<IDynamicModule> modules;
	
	protected ValueImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.VALUE;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Value(this);
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
