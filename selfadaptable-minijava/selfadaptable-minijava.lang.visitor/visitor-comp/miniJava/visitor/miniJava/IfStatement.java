package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface IfStatement extends EObject, AcceptInterface, Statement {
	Expression getExpression();

	void setExpression(Expression value);

	Block getThenBlock();

	void setThenBlock(Block value);

	Block getElseBlock();

	void setElseBlock(Block value);
}
