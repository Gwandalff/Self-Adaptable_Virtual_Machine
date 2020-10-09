package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface WhileStatement extends EObject, Statement {
	Expression getCondition();

	void setCondition(Expression value);

	Block getBlock();

	void setBlock(Block value);

	void evaluateStatement(State state);
}
