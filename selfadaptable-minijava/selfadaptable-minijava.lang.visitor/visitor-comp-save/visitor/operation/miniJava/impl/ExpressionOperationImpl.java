package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;

public class ExpressionOperationImpl extends StatementOperationImpl implements ExpressionOperation {
	private final Expression it;

	private final VisitorInterface vis;

	public ExpressionOperationImpl(Expression it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatement(State state) {
		((ExpressionOperation)this.it.accept(vis)).evaluateExpression((State) (state));
	}

	public Value evaluateExpression(State state) {
		Value result;
		result = ((Value) (null));
		return result;
	}
}
