package miniJava.interpreter.miniJava;

import java.lang.String;
import org.eclipse.emf.ecore.EObject;

public interface Value extends EObject {
	String customToString();

	Value copyj();
}
