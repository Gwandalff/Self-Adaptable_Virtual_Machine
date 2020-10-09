package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.BooleanTypeRef;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;

public class BooleanTypeRefImpl extends SingleTypeRefImpl implements BooleanTypeRef {
	protected BooleanTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.BOOLEAN_TYPE_REF;
	}
}
