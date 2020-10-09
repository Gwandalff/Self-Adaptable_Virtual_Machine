package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.VoidTypeRef;
import org.eclipse.emf.ecore.EClass;

public class VoidTypeRefImpl extends SingleTypeRefImpl implements VoidTypeRef {
	protected VoidTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.VOID_TYPE_REF;
	}
}
