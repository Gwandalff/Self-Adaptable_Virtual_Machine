package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.ForStatement;
import miniJava.visitor.miniJava.State;
import visitor.VisitorInterface;
import visitor.operation.miniJava.AssignmentOperation;
import visitor.operation.miniJava.BlockOperation;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.ForStatementOperation;
import visitor.operation.miniJava.StateOperation;

public class ForStatementOperationImpl extends StatementOperationImpl implements ForStatementOperation {
	private final ForStatement it;

	private final VisitorInterface vis;

	public ForStatementOperationImpl(ForStatement it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatement(State state) {
		((StateOperation)state.accept(vis)).pushNewContext();
		((AssignmentOperation)this.it.getDeclaration().accept(vis)).evaluateStatement((State) (state));
		BooleanValue continueFor = ((BooleanValue) (((BooleanValue) (((ExpressionOperation)this.it.getCondition().accept(vis)).evaluateExpression((State) (state))))));
		while (continueFor.isValue()) {
			((BlockOperation)this.it.getBlock().accept(vis)).evaluateStatement((State) (state));
			((AssignmentOperation)this.it.getProgression().accept(vis)).evaluateStatement((State) (state));
			BooleanValue continueFor2 = ((BooleanValue) (((BooleanValue) (((ExpressionOperation)this.it.getCondition().accept(vis)).evaluateExpression((State) (state))))));
			continueFor = ((BooleanValue) (continueFor2));
		}
		((StateOperation)state.accept(vis)).popCurrentContext();
	}
}
