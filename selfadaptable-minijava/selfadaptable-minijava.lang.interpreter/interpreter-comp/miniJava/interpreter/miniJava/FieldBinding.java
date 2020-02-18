package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface FieldBinding extends EObject {
	Field getField();

	void setField(Field value);

	Value getValue();

	void setValue(Value value);
}
