package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Super;
import org.eclipse.emf.ecore.EClass;

public class SuperImpl extends ExpressionImpl implements Super {
	protected SuperImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.SUPER;
	}
}
