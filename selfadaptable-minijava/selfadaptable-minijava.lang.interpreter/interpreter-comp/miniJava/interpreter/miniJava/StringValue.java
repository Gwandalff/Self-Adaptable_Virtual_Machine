package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface StringValue extends EObject, Value {
	String getValue();

	void setValue(String value);

	String customToString();

	Value copyj();
}
