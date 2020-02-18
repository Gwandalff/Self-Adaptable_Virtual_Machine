package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Or;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.OrOperation;

public class OrOperationImpl extends ExpressionOperationImpl implements OrOperation {
	private final Or it;

	private final VisitorInterface vis;

	public OrOperationImpl(Or it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		BooleanValue left = ((BooleanValue) (((ExpressionOperation)this.it.getLeft().accept(vis)).evaluateExpression((State) (state))));
		BooleanValue right = ((BooleanValue) (((ExpressionOperation)this.it.getRight().accept(vis)).evaluateExpression((State) (state))));
		BooleanValue res = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
		res.setValue(((left.isValue()) || (right.isValue())));
		result = ((BooleanValue) (res));
		return result;
	}
}
