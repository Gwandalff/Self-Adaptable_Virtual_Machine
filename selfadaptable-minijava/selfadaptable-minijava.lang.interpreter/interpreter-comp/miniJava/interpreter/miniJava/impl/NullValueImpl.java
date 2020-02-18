package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.NullValue;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.ecore.EClass;

public class NullValueImpl extends ValueImpl implements NullValue {
	protected NullValueImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.NULL_VALUE;
	}

	public Value copyj() {
		Value result;
		NullValue tmp = ((NullValue) (MiniJavaFactory.eINSTANCE.createNullValue()));
		result = (Value) (tmp) ;
		return result;
	}
}
