package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Clazz extends EObject, AcceptInterface, TypeDeclaration {
	boolean isIsabstract();

	void setIsabstract(boolean value);

	Clazz getSuperClass();

	void setSuperClass(Clazz value);
}
