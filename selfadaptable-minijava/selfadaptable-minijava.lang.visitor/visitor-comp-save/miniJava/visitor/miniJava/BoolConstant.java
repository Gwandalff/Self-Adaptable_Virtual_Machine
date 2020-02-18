package miniJava.visitor.miniJava;

import java.lang.String;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface BoolConstant extends EObject, AcceptInterface, Expression {
	String getValue();

	void setValue(String value);
}
