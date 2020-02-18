package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Call;
import visitor.VisitorInterface;
import visitor.operation.miniJava.CallOperation;

public abstract class CallOperationImpl implements CallOperation {
	private final Call it;

	private final VisitorInterface vis;

	public CallOperationImpl(Call it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
