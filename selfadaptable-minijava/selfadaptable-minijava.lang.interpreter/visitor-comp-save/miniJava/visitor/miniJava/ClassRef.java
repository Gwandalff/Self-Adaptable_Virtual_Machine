package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ClassRef extends EObject, AcceptInterface, SingleTypeRef {
	TypeDeclaration getReferencedClass();

	void setReferencedClass(TypeDeclaration value);
}
