package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.NewCall;
import visitor.VisitorInterface;
import visitor.operation.miniJava.NewCallOperation;

public class NewCallOperationImpl extends CallOperationImpl implements NewCallOperation {
	private final NewCall it;

	private final VisitorInterface vis;

	public NewCallOperationImpl(NewCall it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
