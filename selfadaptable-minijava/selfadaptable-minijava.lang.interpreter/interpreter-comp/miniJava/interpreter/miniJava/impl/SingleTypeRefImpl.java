package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.SingleTypeRef;
import org.eclipse.emf.ecore.EClass;

public class SingleTypeRefImpl extends TypeRefImpl implements SingleTypeRef {
	protected SingleTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.SINGLE_TYPE_REF;
	}
}
