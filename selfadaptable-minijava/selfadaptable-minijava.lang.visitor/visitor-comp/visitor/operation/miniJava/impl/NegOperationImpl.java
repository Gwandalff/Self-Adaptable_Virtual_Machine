package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Neg;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.NegOperation;

public class NegOperationImpl extends ExpressionOperationImpl implements NegOperation {
	private final Neg it;

	private final VisitorInterface vis;

	public NegOperationImpl(Neg it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		IntegerValue left = ((IntegerValue) (((ExpressionOperation)this.it.getExpression().accept(vis)).evaluateExpression((State) (state))));
		IntegerValue tmp = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
		tmp.setValue(-(left.getValue()));
		result = ((IntegerValue) (tmp));
		return result;
	}
}
