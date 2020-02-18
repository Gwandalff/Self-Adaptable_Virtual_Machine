package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface IfStatement extends EObject, Statement {
	Expression getExpression();

	void setExpression(Expression value);

	Block getThenBlock();

	void setThenBlock(Block value);

	Block getElseBlock();

	void setElseBlock(Block value);

	void evaluateStatement(State state);
}
