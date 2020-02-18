package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ArrayTypeRef extends EObject, AcceptInterface, TypeRef {
	SingleTypeRef getTypeRef();

	void setTypeRef(SingleTypeRef value);
}
