package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Clazz;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ClazzOperation;

public class ClazzOperationImpl extends TypeDeclarationOperationImpl implements ClazzOperation {
	private final Clazz it;

	private final VisitorInterface vis;

	public ClazzOperationImpl(Clazz it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
