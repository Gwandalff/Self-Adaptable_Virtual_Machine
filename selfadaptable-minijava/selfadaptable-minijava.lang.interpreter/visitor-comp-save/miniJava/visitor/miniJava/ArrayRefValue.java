package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ArrayRefValue extends EObject, AcceptInterface, Value {
	ArrayInstance getInstance();

	void setInstance(ArrayInstance value);
}
