package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.IntegerTypeRef;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;

public class IntegerTypeRefImpl extends SingleTypeRefImpl implements IntegerTypeRef {
	protected IntegerTypeRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.INTEGER_TYPE_REF;
	}
}
