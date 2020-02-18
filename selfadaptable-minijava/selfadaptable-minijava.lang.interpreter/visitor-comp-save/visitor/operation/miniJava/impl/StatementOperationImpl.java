package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Statement;
import visitor.VisitorInterface;
import visitor.operation.miniJava.StatementOperation;

public class StatementOperationImpl implements StatementOperation {
	private final Statement it;

	private final VisitorInterface vis;

	public StatementOperationImpl(Statement it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatement(State state) {
	}
}
