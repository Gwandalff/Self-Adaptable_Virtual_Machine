package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface ClassRef extends EObject, SingleTypeRef {
	TypeDeclaration getReferencedClass();

	void setReferencedClass(TypeDeclaration value);

	boolean compare(TypeRef other);
}
