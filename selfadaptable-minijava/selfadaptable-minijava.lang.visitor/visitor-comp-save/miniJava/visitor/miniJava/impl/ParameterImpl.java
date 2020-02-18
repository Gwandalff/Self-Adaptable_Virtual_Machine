package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Parameter;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class ParameterImpl extends SymbolImpl implements Parameter {
	protected ParameterImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.PARAMETER;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Parameter(this);
	}
}
