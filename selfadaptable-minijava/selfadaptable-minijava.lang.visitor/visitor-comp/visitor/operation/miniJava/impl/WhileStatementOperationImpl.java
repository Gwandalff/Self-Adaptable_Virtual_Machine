package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.WhileStatement;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BlockOperation;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.WhileStatementOperation;

public class WhileStatementOperationImpl extends StatementOperationImpl implements WhileStatementOperation {
	private final WhileStatement it;

	private final VisitorInterface vis;

	public WhileStatementOperationImpl(WhileStatement it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatement(State state) {
		BooleanValue continueWhile = ((BooleanValue) (((BooleanValue) (((ExpressionOperation)this.it.getCondition().accept(vis)).evaluateExpression((State) (state))))));
		while (continueWhile.isValue()) {
			((BlockOperation)this.it.getBlock().accept(vis)).evaluateStatement((State) (state));
			BooleanValue continueWhile2 = ((BooleanValue) (((BooleanValue) (((ExpressionOperation)this.it.getCondition().accept(vis)).evaluateExpression((State) (state))))));
			continueWhile = ((BooleanValue) (continueWhile2));
		}
	}
}
