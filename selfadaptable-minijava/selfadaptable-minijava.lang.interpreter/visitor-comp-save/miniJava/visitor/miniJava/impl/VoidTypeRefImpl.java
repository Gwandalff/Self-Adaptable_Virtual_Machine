package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.VoidTypeRef;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class VoidTypeRefImpl extends SingleTypeRefImpl implements VoidTypeRef {
	protected VoidTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.VOID_TYPE_REF;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__VoidTypeRef(this);
	}
}
