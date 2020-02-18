package miniJava.visitor.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ArrayInstance extends EObject, AcceptInterface {
	int getSize();

	void setSize(int value);

	EList<Value> getValue();
}
