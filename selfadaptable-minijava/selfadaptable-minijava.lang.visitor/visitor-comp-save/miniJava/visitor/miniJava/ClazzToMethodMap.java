package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ClazzToMethodMap extends EObject, AcceptInterface {
	Clazz getKey();

	void setKey(Clazz value);

	Method getValue();

	void setValue(Method value);
}
