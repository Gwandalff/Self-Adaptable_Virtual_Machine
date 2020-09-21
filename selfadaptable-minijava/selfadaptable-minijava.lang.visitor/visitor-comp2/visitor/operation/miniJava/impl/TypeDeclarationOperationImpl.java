package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.TypeDeclaration;
import visitor.VisitorInterface;
import visitor.operation.miniJava.TypeDeclarationOperation;

public class TypeDeclarationOperationImpl extends NamedElementOperationImpl implements TypeDeclarationOperation {
	private final TypeDeclaration it;

	private final VisitorInterface vis;

	public TypeDeclarationOperationImpl(TypeDeclaration it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
