package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.This;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class ThisImpl extends ExpressionImpl implements This {
	protected ThisImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.THIS;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__This(this);
	}
}
