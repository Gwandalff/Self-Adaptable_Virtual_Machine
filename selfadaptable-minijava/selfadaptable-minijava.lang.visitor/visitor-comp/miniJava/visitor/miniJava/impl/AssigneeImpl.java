package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.HashSet;
import java.util.Set;

import miniJava.visitor.miniJava.Assignee;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class AssigneeImpl extends MinimalEObjectImpl.Container implements Assignee {
	protected Set<IDynamicModule> modules;
	
	protected AssigneeImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.ASSIGNEE;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Assignee(this);
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
