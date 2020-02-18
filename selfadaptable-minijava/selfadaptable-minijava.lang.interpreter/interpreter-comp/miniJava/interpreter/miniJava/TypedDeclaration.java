package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface TypedDeclaration extends EObject, NamedElement {
	TypeRef getTypeRef();

	void setTypeRef(TypeRef value);
}
