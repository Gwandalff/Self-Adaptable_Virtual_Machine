package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface TypedDeclaration extends EObject, AcceptInterface, NamedElement {
	TypeRef getTypeRef();

	void setTypeRef(TypeRef value);
}
