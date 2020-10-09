package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface ForStatement extends EObject, Statement {
	Assignment getDeclaration();

	void setDeclaration(Assignment value);

	Expression getCondition();

	void setCondition(Expression value);

	Assignment getProgression();

	void setProgression(Assignment value);

	Block getBlock();

	void setBlock(Block value);

	void evaluateStatement(State state);
}
