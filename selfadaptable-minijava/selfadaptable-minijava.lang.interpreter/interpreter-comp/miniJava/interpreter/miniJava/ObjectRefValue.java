package miniJava.interpreter.miniJava;

import java.lang.String;
import org.eclipse.emf.ecore.EObject;

public interface ObjectRefValue extends EObject, Value {
	ObjectInstance getInstance();

	void setInstance(ObjectInstance value);

	String customToString();

	Value copyj();
}
