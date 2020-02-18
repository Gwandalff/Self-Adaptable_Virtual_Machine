package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Assignee;
import visitor.VisitorInterface;
import visitor.operation.miniJava.AssigneeOperation;

public class AssigneeOperationImpl implements AssigneeOperation {
	private final Assignee it;

	private final VisitorInterface vis;

	public AssigneeOperationImpl(Assignee it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
