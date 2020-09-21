package miniJava.visitor.miniJava;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface NewObject extends EObject, AcceptInterface, Expression {
	Clazz getType();

	void setType(Clazz value);

	EList<Expression> getArgs();
}
