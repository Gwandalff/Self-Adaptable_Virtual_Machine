package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Super;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class SuperImpl extends ExpressionImpl implements Super {
	protected SuperImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.SUPER;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Super(this);
	}
}
