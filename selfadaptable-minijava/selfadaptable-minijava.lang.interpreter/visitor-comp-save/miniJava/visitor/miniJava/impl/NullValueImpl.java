package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.NullValue;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class NullValueImpl extends ValueImpl implements NullValue {
	protected NullValueImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.NULL_VALUE;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__NullValue(this);
	}
}
