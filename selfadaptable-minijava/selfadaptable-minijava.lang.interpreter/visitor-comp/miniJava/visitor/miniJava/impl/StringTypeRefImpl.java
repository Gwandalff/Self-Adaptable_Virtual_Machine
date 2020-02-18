package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.StringTypeRef;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class StringTypeRefImpl extends SingleTypeRefImpl implements StringTypeRef {
	protected StringTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.STRING_TYPE_REF;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__StringTypeRef(this);
	}
}
