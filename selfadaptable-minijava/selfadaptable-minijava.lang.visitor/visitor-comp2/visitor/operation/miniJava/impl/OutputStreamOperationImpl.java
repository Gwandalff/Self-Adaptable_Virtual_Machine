package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.OutputStream;
import visitor.VisitorInterface;
import visitor.operation.miniJava.OutputStreamOperation;

public class OutputStreamOperationImpl implements OutputStreamOperation {
	private final OutputStream it;

	private final VisitorInterface vis;

	public OutputStreamOperationImpl(OutputStream it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
