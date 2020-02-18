package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Statement;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class StatementImpl extends MinimalEObjectImpl.Container implements Statement {
	protected StatementImpl() {
		super();
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
		// TODO Auto-generated method stub
		
	}

	@Override
	public void detach(IDynamicModule dm) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void notifyDynamicModules() {
		// TODO Auto-generated method stub
		
	}
}
