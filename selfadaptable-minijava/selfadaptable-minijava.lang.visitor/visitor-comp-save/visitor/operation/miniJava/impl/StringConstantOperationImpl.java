package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.StringConstant;
import miniJava.visitor.miniJava.StringValue;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.StringConstantOperation;

public class StringConstantOperationImpl extends ExpressionOperationImpl implements StringConstantOperation {
	private final StringConstant it;

	private final VisitorInterface vis;

	public StringConstantOperationImpl(StringConstant it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		StringValue ret = ((StringValue) (MiniJavaFactory.eINSTANCE.createStringValue()));
		ret.setValue(this.it.getValue());
		result = ((StringValue) (ret));
		return result;
	}
}
