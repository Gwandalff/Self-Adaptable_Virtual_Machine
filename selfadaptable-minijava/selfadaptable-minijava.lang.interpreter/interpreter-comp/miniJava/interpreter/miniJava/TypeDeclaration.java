package miniJava.interpreter.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;

public interface TypeDeclaration extends EObject, NamedElement {
	AccessLevel getAccessLevel();

	void setAccessLevel(AccessLevel value);

	EList<Interface> getImplementz();

	EList<Member> getMembers();
}
