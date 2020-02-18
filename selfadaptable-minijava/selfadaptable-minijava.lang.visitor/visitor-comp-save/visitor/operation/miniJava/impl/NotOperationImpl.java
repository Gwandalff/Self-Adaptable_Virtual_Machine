package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Not;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.NotOperation;

public class NotOperationImpl extends ExpressionOperationImpl implements NotOperation {
	private final Not it;

	private final VisitorInterface vis;

	public NotOperationImpl(Not it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		BooleanValue left = ((BooleanValue) (((ExpressionOperation)this.it.getExpression().accept(vis)).evaluateExpression((State) (state))));
		BooleanValue tmp = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
		tmp.setValue(!(left.isValue()));
		result = ((BooleanValue) (tmp));
		return result;
	}
}
