package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Member extends EObject, AcceptInterface, TypedDeclaration {
	AccessLevel getAccess();

	void setAccess(AccessLevel value);
}
