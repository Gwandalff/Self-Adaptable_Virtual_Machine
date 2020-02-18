package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.IntConstant;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.IntConstantOperation;

public class IntConstantOperationImpl extends ExpressionOperationImpl implements IntConstantOperation {
	private final IntConstant it;

	private final VisitorInterface vis;

	public IntConstantOperationImpl(IntConstant it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		IntegerValue ret = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
		ret.setValue(this.it.getValue());
		result = ((IntegerValue) (ret));
		return result;
	}
}
