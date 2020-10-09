package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Field extends EObject, Member {
	Expression getDefaultValue();

	void setDefaultValue(Expression value);
}
