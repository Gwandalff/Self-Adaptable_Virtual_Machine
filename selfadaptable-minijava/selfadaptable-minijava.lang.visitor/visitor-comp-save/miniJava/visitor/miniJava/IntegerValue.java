package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface IntegerValue extends EObject, AcceptInterface, Value {
	int getValue();

	void setValue(int value);
}
