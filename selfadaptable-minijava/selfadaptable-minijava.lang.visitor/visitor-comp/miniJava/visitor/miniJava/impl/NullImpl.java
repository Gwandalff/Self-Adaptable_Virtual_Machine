package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Null;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class NullImpl extends ExpressionImpl implements Null {
	protected NullImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.NULL;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Null(this);
	}
}
