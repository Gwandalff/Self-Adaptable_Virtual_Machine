package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface ForStatement extends EObject, AcceptInterface, Statement {
	Assignment getDeclaration();

	void setDeclaration(Assignment value);

	Expression getCondition();

	void setCondition(Expression value);

	Assignment getProgression();

	void setProgression(Assignment value);

	Block getBlock();

	void setBlock(Block value);
}
