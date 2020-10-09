package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Return extends EObject, Statement {
	Expression getExpression();

	void setExpression(Expression value);

	void evaluateStatement(State state);
}
