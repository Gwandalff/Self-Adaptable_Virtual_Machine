package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.TypedDeclaration;
import visitor.VisitorInterface;
import visitor.operation.miniJava.TypedDeclarationOperation;

public class TypedDeclarationOperationImpl extends NamedElementOperationImpl implements TypedDeclarationOperation {
	private final TypedDeclaration it;

	private final VisitorInterface vis;

	public TypedDeclarationOperationImpl(TypedDeclaration it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
