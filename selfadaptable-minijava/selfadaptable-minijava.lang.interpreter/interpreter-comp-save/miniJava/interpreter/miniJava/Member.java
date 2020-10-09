package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Member extends EObject, TypedDeclaration {
	AccessLevel getAccess();

	void setAccess(AccessLevel value);
}
