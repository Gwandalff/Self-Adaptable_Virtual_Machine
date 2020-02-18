package miniJava.visitor.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface TypeDeclaration extends EObject, AcceptInterface, NamedElement {
	AccessLevel getAccessLevel();

	void setAccessLevel(AccessLevel value);

	EList<Interface> getImplementz();

	EList<Member> getMembers();
}
