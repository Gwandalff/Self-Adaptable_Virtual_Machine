package miniJava.interpreter.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;

public interface ObjectInstance extends EObject {
	EList<FieldBinding> getFieldbindings();

	Clazz getType();

	void setType(Clazz value);
}
