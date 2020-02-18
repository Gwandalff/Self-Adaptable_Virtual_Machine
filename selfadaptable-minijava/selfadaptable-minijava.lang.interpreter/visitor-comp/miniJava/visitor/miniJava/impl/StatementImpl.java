package miniJava.visitor.miniJava.impl;

import java.util.HashSet;
import java.util.Set;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Statement;
import visitor.IDynamicModule;
import visitor.VisitorInterface;

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

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Statement(this);
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
