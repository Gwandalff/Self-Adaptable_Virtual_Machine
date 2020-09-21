package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.SingleTypeRef;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class SingleTypeRefImpl extends TypeRefImpl implements SingleTypeRef {
	protected SingleTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.SINGLE_TYPE_REF;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__SingleTypeRef(this);
	}
}
