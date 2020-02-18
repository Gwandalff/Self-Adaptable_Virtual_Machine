package miniJava.interpreter.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;

public interface ArrayInstance extends EObject {
	int getSize();

	void setSize(int value);

	EList<Value> getValue();
}
