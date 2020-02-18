package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Import;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ImportOperation;

public class ImportOperationImpl implements ImportOperation {
	private final Import it;

	private final VisitorInterface vis;

	public ImportOperationImpl(Import it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}
}
