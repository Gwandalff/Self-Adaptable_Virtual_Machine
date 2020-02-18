package miniJava.visitor.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ObjectInstance extends EObject, AcceptInterface {
	EList<FieldBinding> getFieldbindings();

	Clazz getType();

	void setType(Clazz value);
}
