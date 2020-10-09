package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface ArrayTypeRef extends EObject, SingleTypeRef {
	SingleTypeRef getTypeRef();

	void setTypeRef(SingleTypeRef value);
}
