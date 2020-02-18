package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.ObjectRefValue;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.This;
import miniJava.visitor.miniJava.Value;
import visitor.VisitorInterface;
import visitor.operation.miniJava.StateOperation;
import visitor.operation.miniJava.ThisOperation;

public class ThisOperationImpl extends ExpressionOperationImpl implements ThisOperation {
	private final This it;

	private final VisitorInterface vis;

	public ThisOperationImpl(This it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		ObjectInstance currentInstance = ((ObjectInstance) (((StateOperation)state.accept(vis)).findCurrentFrame().getInstance()));
		ObjectRefValue tmp = ((ObjectRefValue) (MiniJavaFactory.eINSTANCE.createObjectRefValue()));
		tmp.setInstance(currentInstance);
		result = ((ObjectRefValue) (tmp));
		return result;
	}
}
