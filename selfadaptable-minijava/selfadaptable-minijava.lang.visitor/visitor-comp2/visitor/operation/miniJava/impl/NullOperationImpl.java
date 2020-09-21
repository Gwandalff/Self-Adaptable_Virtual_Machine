package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Null;
import miniJava.visitor.miniJava.NullValue;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.NullOperation;

public class NullOperationImpl extends ExpressionOperationImpl implements NullOperation {
	private final Null it;

	private final VisitorInterface vis;

	public NullOperationImpl(Null it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		result = ((NullValue) (MiniJavaFactory.eINSTANCE.createNullValue()));
		return result;
	}
}
