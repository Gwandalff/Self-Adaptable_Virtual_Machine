package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface TypeRef extends EObject {
	boolean compare(TypeRef other);
}
