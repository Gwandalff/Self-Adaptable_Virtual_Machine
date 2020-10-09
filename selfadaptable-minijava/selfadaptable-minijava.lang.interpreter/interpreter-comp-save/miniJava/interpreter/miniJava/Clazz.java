package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Clazz extends EObject, TypeDeclaration {
	boolean isIsabstract();

	void setIsabstract(boolean value);

	Clazz getSuperClass();

	void setSuperClass(Clazz value);
}
