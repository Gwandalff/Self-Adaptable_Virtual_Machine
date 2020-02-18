package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.IntegerTypeRef;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class IntegerTypeRefImpl extends SingleTypeRefImpl implements IntegerTypeRef {
	protected IntegerTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.INTEGER_TYPE_REF;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__IntegerTypeRef(this);
	}
}
