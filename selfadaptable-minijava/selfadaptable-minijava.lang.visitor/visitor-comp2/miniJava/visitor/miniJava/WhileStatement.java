package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface WhileStatement extends EObject, AcceptInterface, Statement {
	Expression getCondition();

	void setCondition(Expression value);

	Block getBlock();

	void setBlock(Block value);
}
