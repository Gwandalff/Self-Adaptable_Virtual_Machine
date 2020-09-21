package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.BooleanTypeRef;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class BooleanTypeRefImpl extends SingleTypeRefImpl implements BooleanTypeRef {
	protected BooleanTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.BOOLEAN_TYPE_REF;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__BooleanTypeRef(this);
	}
}
