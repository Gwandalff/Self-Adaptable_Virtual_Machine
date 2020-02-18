package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface FieldBinding extends EObject, AcceptInterface {
	Field getField();

	void setField(Field value);

	Value getValue();

	void setValue(Value value);
}
