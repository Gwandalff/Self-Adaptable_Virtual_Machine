package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.ClazzToMethodMap;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ClazzToMethodMapOperation;

public class ClazzToMethodMapOperationImpl implements ClazzToMethodMapOperation {
	private final ClazzToMethodMap it;

	private final VisitorInterface vis;

	public ClazzToMethodMapOperationImpl(ClazzToMethodMap it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
