package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ObjectRefValue extends EObject, AcceptInterface, Value {
	ObjectInstance getInstance();

	void setInstance(ObjectInstance value);
}
