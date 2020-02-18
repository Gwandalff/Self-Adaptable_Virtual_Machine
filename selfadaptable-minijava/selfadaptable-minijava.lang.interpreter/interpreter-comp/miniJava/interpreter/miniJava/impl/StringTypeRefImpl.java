package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.StringTypeRef;
import org.eclipse.emf.ecore.EClass;

public class StringTypeRefImpl extends SingleTypeRefImpl implements StringTypeRef {
	protected StringTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.STRING_TYPE_REF;
	}
}
